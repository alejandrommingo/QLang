"""
Step 1 of the methodological pathway: corpus loading.

Three input paths, one single output format:

  - load_from_wikipedia(...)        -> predefined corpus, via the web API
  - sample_wikipedia_dataset(...)   -> large-scale sampling via HF dataset
  - load_from_files(...)            -> corpus to be defined (project texts)

All of them produce the SAME common format (Corpus), so that every later
phase (target unit, sampling, offsets...) operates on a single structure and
no logic has to be duplicated.

Common format (JSON-serializable):

    {
        "doc_001": {
            "body": "document text...",
            "meta": {"source": "wikipedia", "language": "en", ...}
        },
        ...
    }

Access is by name (doc["body"], doc["meta"]["source"]), not by position, so
it is robust and self-documenting.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import requests


# ---------------------------------------------------------------------------
# Common format
# ---------------------------------------------------------------------------

# A document is a dict with 'body' (str) and 'meta' (dict). A corpus is a dict
# {doc_id: document}. We use type aliases so later function signatures read
# clearly.
Document = Dict[str, Any]
Corpus = Dict[str, Document]


def make_document(body: str, meta: Optional[Dict[str, Any]] = None) -> Document:
    """Build a document in the common format.

    This is the ONLY way documents are created in the pipeline, so we
    guarantee they all share the same structure, wherever they come from.
    """
    if not isinstance(body, str):
        raise TypeError(f"'body' must be str, not {type(body).__name__}")
    return {"body": body, "meta": dict(meta) if meta else {}}


def validate_corpus(corpus: Corpus) -> None:
    """Check that a corpus respects the common format.

    Raises ValueError with a clear message if something is off. Meant to be
    used as a cheap check at the end of any loader.
    """
    if not isinstance(corpus, dict):
        raise ValueError("The corpus must be a dict {id: document}")
    for doc_id, doc in corpus.items():
        if not isinstance(doc_id, str):
            raise ValueError(f"The id {doc_id!r} is not a str")
        if not isinstance(doc, dict):
            raise ValueError(f"Document '{doc_id}' is not a dict")
        if "body" not in doc:
            raise ValueError(f"Document '{doc_id}' has no 'body' key")
        if not isinstance(doc["body"], str):
            raise ValueError(f"'body' of '{doc_id}' is not a str")
        if not isinstance(doc.get("meta", {}), dict):
            raise ValueError(f"'meta' of '{doc_id}' is not a dict")


def save_corpus(corpus: Corpus, path: str, stage: str = "") -> None:
    """Serialize the corpus to JSON with a count summary at the top.

    File shape:
        {"summary": {n_articles, stage}, "data": {id: {body, meta}}}
    so the number of articles is visible at a glance.
    """
    validate_corpus(corpus)
    wrapper = {
        "summary": {"stage": stage, "n_articles": len(corpus)},
        "data": corpus,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)


def load_corpus(path: str) -> Corpus:
    """Load a corpus from JSON. Accepts the new format ({summary, data}) and
    the old one (a bare corpus)."""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, dict) and "data" in content \
            and "summary" in content:
        corpus = content["data"]
    else:
        corpus = content
    validate_corpus(corpus)
    return corpus


# ---------------------------------------------------------------------------
# Path 1: predefined corpus (Wikipedia web API)
# ---------------------------------------------------------------------------

_WIKI_API = "https://{lang}.wikipedia.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent with a contact form; a generic
# one may be rejected with a 403 error. Adjust the email/URL if you like.
_USER_AGENT = ("corpus-loader/0.2 (academic research; "
               "https://example.org; contact@example.org)")


def fetch_wikipedia_article(title: str, lang: str = "en",
                            timeout: int = 10,
                            retries: int = 4) -> Optional[Document]:
    """Download a single Wikipedia article as a Document.

    Returns None if the article does not exist. If Wikipedia answers 429 (Too
    Many Requests) or 503, it waits and retries with growing back-off instead
    of failing. Small, isolated function: one request, one document.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,       # plain text, no HTML
        "redirects": 1,         # follow redirects
        "titles": title,
    }
    for attempt in range(retries):
        resp = requests.get(_WIKI_API.format(lang=lang), params=params,
                            timeout=timeout,
                            headers={"User-Agent": _USER_AGENT})
        # 429 = too many requests; 503 = server busy -> wait and retry
        if resp.status_code in (429, 503):
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"    (Wikipedia asks to wait; retrying in {wait:.0f}s...)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "missing" in page:
                return None
            return make_document(
                body=page.get("extract", ""),
                meta={
                    "source": "wikipedia",
                    "via": "api",
                    "language": lang,
                    "title": page.get("title", title),
                    "page_id": page_id,
                },
            )
        return None
    raise RuntimeError(
        f"Wikipedia keeps rate-limiting (429) after {retries} attempts while "
        f"downloading '{title}'. Try fewer articles or wait a moment.")


def load_from_wikipedia(titles: Iterable[str], lang: str = "en",
                        skip_missing: bool = True,
                        pause: float = 0.3) -> Corpus:
    """Build a corpus from a list of Wikipedia titles.

    Iterates over fetch_wikipedia_article and assembles the common format.
    Each document's id is its normalized title; if an article is missing it
    is skipped (skip_missing=True) or raises. 'pause' is the wait in seconds
    between downloads, to avoid overloading the API (prevents the 429 error).
    """
    corpus: Corpus = {}
    titles = list(titles)
    for i, title in enumerate(titles):
        doc = fetch_wikipedia_article(title, lang=lang)
        if i < len(titles) - 1:
            time.sleep(pause)
        if doc is None:
            if skip_missing:
                continue
            raise ValueError(f"Article not found: {title!r}")
        doc_id = doc["meta"]["title"]
        corpus[doc_id] = doc
    validate_corpus(corpus)
    return corpus


# ---------------------------------------------------------------------------
# Path 2: corpus to be defined (project texts, from files)
# ---------------------------------------------------------------------------

def inspect_file_tree(folder: str, extension: str = ".txt") -> Dict[str, Any]:
    """Describe the files that would be loaded from a corpus folder.

    The exe uses this before loading a local corpus so the user can see how
    many folder levels the corpus has and decide which path levels should
    become metadata keys.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    paths = _collect_file_paths(folder, extension=extension, recursive=True)
    folder_depths = []
    folder_values: List[set] = []
    for path in paths:
        rel_path = os.path.relpath(path, folder)
        folder_parts = _relative_folder_parts(rel_path)
        folder_depths.append(len(folder_parts))
        for i, value in enumerate(folder_parts):
            while len(folder_values) <= i:
                folder_values.append(set())
            folder_values[i].add(value)

    return {
        "n_files": len(paths),
        "min_folder_depth": min(folder_depths) if folder_depths else 0,
        "max_folder_depth": max(folder_depths) if folder_depths else 0,
        "examples": [os.path.relpath(p, folder) for p in paths[:5]],
        "folder_values_by_level": [
            sorted(values)[:8] for values in folder_values
        ],
    }


def load_from_files(folder: str, extension: str = ".txt",
                    encoding: str = "utf-8",
                    meta_per_document: Optional[Dict[str, Dict]] = None,
                    recursive: bool = False,
                    folder_metadata_keys: Optional[Iterable[str]] = None,
                    file_metadata_key: Optional[str] = None,
                    source_from_metadata_key: Optional[str] = None,
                    metadata_from_path: bool = False
                    ) -> Corpus:
    """Build a corpus by reading every file in a folder.

    By default this reads only the files directly inside 'folder', keeping the
    original behavior: each document's id is the file name without extension.

    If recursive=True, it reads matching files in all subfolders and uses the
    relative path without extension as the id, with path separators replaced by
    "__". This avoids collisions such as several folders containing
    article_1.txt.

    Folder and file names can be promoted to metadata in a corpus-agnostic
    way:

      - folder_metadata_keys gives one metadata key per folder level.
      - file_metadata_key stores the file stem, without extension.
      - source_from_metadata_key copies one metadata value into 'source', so
        the existing source filter and stratified sampling can use it.

    Example:

        folder_metadata_keys=["year", "newspaper"],
        file_metadata_key="article_id",
        source_from_metadata_key="newspaper"

    for a file such as 2018/ElPais/article_1.txt adds:

        {"year": "2018", "newspaper": "ElPais", "article_id": "article_1"}

    You can also attach per-document metadata via 'meta_per_document', a dict
    {id: {key: value}}, to document source, period, author, etc.

    metadata_from_path=True is kept for backward compatibility with the old
    year/newspaper/article layout. Prefer the generic metadata-key arguments
    for new corpora.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder does not exist: {folder}")
    meta_per_document = meta_per_document or {}
    if metadata_from_path:
        source_from_metadata_key = source_from_metadata_key or "newspaper"

    corpus: Corpus = {}
    paths = _collect_file_paths(folder, extension=extension,
                                recursive=recursive)

    for path in sorted(paths):
        rel_path = os.path.relpath(path, folder)
        rel_no_ext = os.path.splitext(rel_path)[0]
        doc_id = rel_no_ext.replace(os.sep, "__")
        with open(path, "r", encoding=encoding) as f:
            body = f.read()
        meta = {
            "source": "project",
            "corpus_source": "files",
            "file": rel_path,
        }
        if metadata_from_path:
            meta.update(_metadata_from_year_newspaper_path(rel_path))
        else:
            meta.update(_metadata_from_relative_path(
                rel_path,
                folder_metadata_keys=folder_metadata_keys,
                file_metadata_key=file_metadata_key,
            ))
        if source_from_metadata_key and source_from_metadata_key in meta:
            meta["source"] = meta[source_from_metadata_key]
        meta.update(meta_per_document.get(doc_id, {}))
        corpus[doc_id] = make_document(body=body, meta=meta)
    validate_corpus(corpus)
    return corpus


def load_from_single_file(
        path: str,
        document_separator: str,
        encoding: str = "utf-8",
        has_metadata: bool = False,
        metadata_body_separator: str = "",
        metadata_field_separator: str = "",
        metadata_key_value_separator: str = "=",
        source_from_metadata_key: Optional[str] = None) -> Corpus:
    """Build a corpus from one text file containing several documents.

    'document_separator' splits the file into documents. If has_metadata is
    False, every chunk is treated directly as a document body.

    If has_metadata is True, each chunk must contain a metadata block followed
    by the body. 'metadata_body_separator' splits those two sections.
    'metadata_field_separator' separates metadata fields, and
    'metadata_key_value_separator' separates each metadata key from its value.

    Example document chunk:

        id=doc_1|year=2018|newspaper=ElPais

        Article body...

    with metadata_body_separator="\\n\\n", metadata_field_separator="|", and
    metadata_key_value_separator="=".
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    with open(path, "r", encoding=encoding) as f:
        raw = f.read()

    raw_docs = ([raw] if document_separator == ""
                else raw.split(document_separator))
    corpus: Corpus = {}
    base_name = os.path.splitext(os.path.basename(path))[0]

    for i, raw_doc in enumerate(raw_docs, 1):
        raw_doc = raw_doc.strip()
        if not raw_doc:
            continue
        meta = {
            "source": "project",
            "corpus_source": "single_file",
            "file": os.path.basename(path),
            "document_index": i,
        }
        body = raw_doc
        if has_metadata:
            extra_meta, body = _split_document_metadata(
                raw_doc,
                metadata_body_separator=metadata_body_separator,
                metadata_field_separator=metadata_field_separator,
                metadata_key_value_separator=metadata_key_value_separator,
            )
            meta.update(extra_meta)
        if source_from_metadata_key and source_from_metadata_key in meta:
            meta["source"] = meta[source_from_metadata_key]
        doc_id = _single_file_doc_id(base_name, i, meta, corpus)
        corpus[doc_id] = make_document(body=body.strip(), meta=meta)

    validate_corpus(corpus)
    return corpus


def _collect_file_paths(folder: str, extension: str,
                        recursive: bool) -> List[str]:
    paths = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if name.endswith(extension):
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.endswith(extension):
                paths.append(path)
    return sorted(paths)


def _relative_folder_parts(rel_path: str) -> List[str]:
    folder = os.path.dirname(rel_path)
    return [] if not folder else folder.split(os.sep)


def _metadata_from_relative_path(
        rel_path: str,
        folder_metadata_keys: Optional[Iterable[str]] = None,
        file_metadata_key: Optional[str] = None) -> Dict[str, str]:
    meta = {}
    folder_parts = _relative_folder_parts(rel_path)
    for key, value in zip(folder_metadata_keys or [], folder_parts):
        key = str(key).strip()
        if key:
            meta[key] = value
    if file_metadata_key:
        key = str(file_metadata_key).strip()
        if key:
            meta[key] = os.path.splitext(os.path.basename(rel_path))[0]
    return meta


def _metadata_from_year_newspaper_path(rel_path: str) -> Dict[str, str]:
    parts = rel_path.split(os.sep)
    year_index = next(
        (i for i, part in enumerate(parts[:-2])
         if re.fullmatch(r"\d{4}", part)),
        None,
    )
    if year_index is None:
        return {}
    year, newspaper = parts[year_index], parts[year_index + 1]
    return {
        "year": year,
        "newspaper": newspaper,
        "article_id": os.path.splitext(parts[-1])[0],
    }


def _split_document_metadata(
        raw_doc: str,
        metadata_body_separator: str,
        metadata_field_separator: str,
        metadata_key_value_separator: str) -> tuple:
    if not metadata_body_separator:
        raise ValueError("metadata_body_separator cannot be empty when "
                         "has_metadata=True")
    if metadata_body_separator not in raw_doc:
        raise ValueError("A document does not contain the metadata/body "
                         f"separator {metadata_body_separator!r}")
    raw_meta, body = raw_doc.split(metadata_body_separator, 1)
    return (
        _parse_metadata_fields(
            raw_meta,
            field_separator=metadata_field_separator,
            key_value_separator=metadata_key_value_separator,
        ),
        body,
    )


def _parse_metadata_fields(raw_meta: str, field_separator: str,
                           key_value_separator: str) -> Dict[str, str]:
    fields = ([raw_meta] if not field_separator
              else raw_meta.split(field_separator))
    meta = {}
    positional = 1
    for field in fields:
        field = field.strip()
        if not field:
            continue
        if key_value_separator and key_value_separator in field:
            key, value = field.split(key_value_separator, 1)
            key = key.strip()
            if key:
                meta[key] = value.strip()
            continue
        meta[f"meta_{positional}"] = field
        positional += 1
    return meta


def _single_file_doc_id(base_name: str, index: int, meta: Dict[str, Any],
                        corpus: Corpus) -> str:
    candidate = (
        meta.get("doc_id") or meta.get("id") or meta.get("title")
        or f"{base_name}_{index:04d}"
    )
    candidate = str(candidate).strip() or f"{base_name}_{index:04d}"
    doc_id = re.sub(r"[^\w.-]+", "_", candidate)
    doc_id = doc_id.strip("_") or f"{base_name}_{index:04d}"
    if doc_id not in corpus:
        return doc_id
    n = 2
    while f"{doc_id}_{n}" in corpus:
        n += 1
    return f"{doc_id}_{n}"


# ---------------------------------------------------------------------------
# Wikipedia search (to extend the corpus up to a minimum)
# ---------------------------------------------------------------------------

def search_wikipedia_titles(term: str, lang: str = "en", n: int = 20,
                            timeout: int = 10) -> list:
    """Return titles of Wikipedia articles mentioning the term.

    Uses the search API. Useful to find more articles containing a word when
    the initial corpus falls short.
    """
    params = {
        "action": "query", "format": "json", "list": "search",
        "srsearch": term, "srlimit": n,
    }
    resp = requests.get(_WIKI_API.format(lang=lang), params=params,
                        timeout=timeout,
                        headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    return [r["title"] for r in results]


def search_and_load_wikipedia(term: str, lang: str = "en", n: int = 20,
                              exclude: Optional[set] = None) -> Corpus:
    """Search for articles mentioning the term and download them as a corpus.

    'exclude' is a set of titles not to download again.
    """
    exclude = exclude or set()
    titles = search_wikipedia_titles(term, lang=lang, n=n)
    titles = [t for t in titles if t not in exclude]
    return load_from_wikipedia(titles, lang=lang)


# ===========================================================================
# Path 3: large-scale sampling with the Wikipedia dataset (Hugging Face)
# ===========================================================================
# Unlike the web API (live requests, limited by the 429 error), this path
# streams a full prebuilt Wikipedia dump and samples it locally. It is the
# right one to sample a term across all of Wikipedia. Requires the 'datasets'
# library:  pip install datasets

# dataset configurations of wikimedia/wikipedia by language
_WIKI_DATASET_CFG = {
    "en": "20231101.en",
    "es": "20231101.es",
}

# titles to discard (disambiguations, lists, indexes...)
_JUNK_TITLE = re.compile(
    r"(disambiguation|list of|desambiguación|anexo:|lista de|véase también)",
    flags=re.IGNORECASE)

# typical content of a disambiguation page (even if the title does not say so)
_DISAMBIGUATION = re.compile(
    r"(may refer to|may also refer to|puede referirse a|"
    r"hace referencia a varios)",
    flags=re.IGNORECASE)

# appendix section headers: from here on it is not article body (references
# are already stripped, but "see also", external links, notes, etc. may
# remain). We cut the text at the first one that appears.
_APPENDIX_SECTIONS = re.compile(
    r"\n\s*(See also|References|External links|Further reading|Notes|"
    r"Bibliography|Véase también|Referencias|Enlaces externos|"
    r"Notas|Bibliografía)\s*\n",
    flags=re.IGNORECASE)


def _clean_body(text: str) -> str:
    """Trim the text down to the article body.

    Removes everything from the first appendix header (See also, References,
    Véase también...). The dataset already strips references, but these
    navigation headers may remain.
    """
    m = _APPENDIX_SECTIONS.search(text)
    return text[:m.start()].rstrip() if m else text


def _is_disambiguation(title: str, text: str) -> bool:
    """True if the page looks like a disambiguation (by title or content)."""
    if _JUNK_TITLE.search(title):
        return True
    return bool(_DISAMBIGUATION.search(text[:400]))


def sample_wikipedia_dataset(
        term, lang: str = "en", k: int = 100, seed: int = 123,
        max_iter: Optional[int] = 500000, min_occurrences: int = 2,
        min_chars: int = 1000, max_chars: int = 15000,
        progress_every: int = 0) -> Corpus:
    """Sample k Wikipedia articles containing the term, via the dataset.

    Streams the 'wikimedia/wikipedia' dataset and uses reservoir sampling to
    keep k random articles among those passing the filters (minimum term
    occurrences, minimum/maximum length, and not being junk pages such as
    disambiguations or lists).

    'term' can be a word or a list of words. Returns the corpus in the common
    format {doc_id: {body, meta}}, with metadata including the number of
    occurrences found (hits). If 'progress_every' > 0, prints progress every
    that many scanned articles.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "This path needs the 'datasets' library. "
            "Install it with: pip install datasets")

    terms = [term] if isinstance(term, str) else list(term)
    patterns = [re.compile(r"\b" + re.escape(t.casefold()) + r"\b")
                for t in terms]

    lang_l = lang.lower().strip()
    cfg = _WIKI_DATASET_CFG.get("en" if lang_l.startswith("en")
                                else "es" if lang_l.startswith("es")
                                else None)
    if cfg is None:
        raise ValueError("lang must be 'en' or 'es'.")

    ds = load_dataset("wikimedia/wikipedia", cfg, split="train",
                      streaming=True)

    rng = random.Random(seed)
    reservoir: list = []
    candidates = 0

    for i, ex in enumerate(ds):
        if max_iter is not None and i >= max_iter:
            break
        if progress_every and i and i % progress_every == 0:
            print(f"    scanned {i} articles, {candidates} matches so far...")

        text = ex.get("text", "")
        title = ex.get("title", f"doc_{i}")

        if _is_disambiguation(title, text):
            continue

        text = _clean_body(text)
        if len(text) < min_chars:
            continue

        text_cf = text.casefold()
        hits = sum(len(p.findall(text_cf)) for p in patterns)
        if hits < min_occurrences:
            continue

        candidates += 1
        item = (title, text[:max_chars], hits)
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(1, candidates)
            if j <= k:
                reservoir[j - 1] = item

    corpus: Corpus = {}
    for title, body, hits in reservoir:
        corpus[title] = make_document(
            body=body,
            meta={"source": "wikipedia", "via": "dataset",
                  "language": lang_l[:2], "title": title, "hits": hits})
    validate_corpus(corpus)
    return corpus
