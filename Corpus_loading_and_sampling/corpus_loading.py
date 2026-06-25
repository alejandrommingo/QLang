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
from typing import Any, Dict, Iterable, Optional

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

def load_from_files(folder: str, extension: str = ".txt",
                    encoding: str = "utf-8",
                    meta_per_document: Optional[Dict[str, Dict]] = None
                    ) -> Corpus:
    """Build a corpus by reading every file in a folder.

    Each document's id is the file name without extension. You can attach
    per-document metadata via 'meta_per_document', a dict {id: {key: value}},
    to document source, period, author, etc.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder does not exist: {folder}")
    meta_per_document = meta_per_document or {}

    corpus: Corpus = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(extension):
            continue
        path = os.path.join(folder, name)
        doc_id = os.path.splitext(name)[0]
        with open(path, "r", encoding=encoding) as f:
            body = f.read()
        meta = {"source": "project", "file": name}
        meta.update(meta_per_document.get(doc_id, {}))
        corpus[doc_id] = make_document(body=body, meta=meta)
    validate_corpus(corpus)
    return corpus


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
