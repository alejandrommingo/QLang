"""
Phase 4/5 (static path): Latent Semantic Analysis (LSA).

This is the STATIC, distributional alternative to contextual models. Unlike
BERT/GPT (where a word's vector depends on its sentence), LSA gives ONE vector
per word, the same in every occurrence, built from how words co-occur across
documents. The article contrasts exactly these two families (static vs.
contextual); this module is the static one.

How it works:
  1. Build a term-document matrix: rows = words, columns = documents, each
     cell = how often the word appears in that document (optionally weighted
     by TF-IDF, which downweights words common to all documents).
  2. Reduce it with Truncated SVD to 'n_components' latent dimensions. Each
     word ends up as a dense vector in that reduced space.

Input: the PREPARED units (from text_prep) OR the corpus they came from. LSA
needs the document texts, so it works at the document level: it reads the
prepared sample, recovers the documents involved, and builds the matrix over
them.

Output: a vector per word (at least the target word) plus the model pieces,
saved to JSON, and a record in the TraceLog.

Needs: numpy and scikit-learn (pip install scikit-learn).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tracelog import TraceLog

Unit = Dict[str, Any]


class _NullProgress:
    def __init__(self, desc=""):
        self.desc = desc

    def update(self, n=1):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SimpleProgressBar:
    def __init__(self, total, desc="", unit="step"):
        self.total = max(int(total), 0)
        self.desc = desc or "Progress"
        self.unit = unit
        self.count = 0

    def update(self, n=1):
        self.count = min(self.total, self.count + n)
        pct = 100 if self.total == 0 else int(self.count * 100 / self.total)
        filled = pct // 5
        bar = "#" * filled + "." * (20 - filled)
        end = "\n" if self.count >= self.total else "\r"
        print(f"{self.desc}: [{bar}] {self.count}/{self.total} {self.unit}",
              end=end, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.total and self.count < self.total:
            print()
        return False


def _simple_progress(iterable, total=None, desc="", unit="item"):
    label = desc or "Progress"
    total = int(total) if total is not None else None
    step = max(total // 20, 1) if total else 1000
    for i, item in enumerate(iterable, 1):
        yield item
        if total:
            if i == 1 or i == total or i % step == 0:
                pct = min(100, int(i * 100 / total))
                filled = pct // 5
                bar = "#" * filled + "." * (20 - filled)
                end = "\n" if i >= total else "\r"
                print(f"{label}: [{bar}] {i}/{total} {unit}", end=end,
                      flush=True)
        elif i % step == 0:
            print(f"{label}: {i} {unit}", flush=True)


def _progress(iterable, total=None, desc="", unit="item", enabled=False):
    """Wrap an iterable with tqdm when progress output is requested."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return _simple_progress(iterable, total=total, desc=desc, unit=unit)
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def _progress_bar(total, desc="", unit="step", enabled=False):
    """Create a manual tqdm progress bar when requested."""
    if not enabled:
        return _NullProgress(desc)
    try:
        from tqdm import tqdm
    except ImportError:
        return _SimpleProgressBar(total, desc=desc, unit=unit)
    return tqdm(total=total, desc=desc, unit=unit)


def estimate_lsa(documents: List[str], target_words: Optional[List[str]] = None,
                 n_components: int = 100, use_tfidf: bool = True,
                 min_df: int = 1, seed: int = 0,
                 save_to: Optional[str] = None,
                 log: Optional[TraceLog] = None,
                 justification: str = "",
                 show_progress: bool = False) -> Dict[str, Any]:
    """Estimate an LSA space over a list of document texts.

    Builds a term-document matrix (TF-IDF by default) and reduces it with
    Truncated SVD to 'n_components' dimensions. Returns a dict with the
    vocabulary, the reduced word vectors, and (if given) the vectors of the
    target words pulled out for convenience.

    'min_df' drops words appearing in fewer than that many documents (noise).
    Saves the result and records the decision in the log.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import (
            CountVectorizer, TfidfVectorizer)
        from sklearn.decomposition import TruncatedSVD
    except ImportError as e:
        raise ImportError(
            "LSA needs numpy and scikit-learn. "
            "Install with: pip install scikit-learn") from e

    if len(documents) < 2:
        raise ValueError(
            "LSA needs at least 2 documents to find co-occurrence structure.")

    total_steps = 3 + (1 if save_to is not None else 0)
    with _progress_bar(total_steps, desc="Estimating LSA", unit="step",
                       enabled=show_progress) as progress:
        # 1) term-document matrix (here rows are documents; we transpose
        #    meaning via the SVD on the document-term matrix, which is
        #    equivalent for word vectors taken from the components).
        Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
        vectorizer = Vectorizer(min_df=min_df, lowercase=False)
        dtm = vectorizer.fit_transform(documents)          # docs x terms
        vocab = vectorizer.get_feature_names_out().tolist()
        progress.update(1)

        # 2) reduce. We want WORD vectors, so we factor the term-document
        #    matrix (transpose: terms x docs) and take the reduced term
        #    representation.
        n_comp = min(n_components, min(dtm.shape) - 1)
        if n_comp < 1:
            raise ValueError("Not enough data for the requested n_components.")
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        term_doc = dtm.T                                   # terms x docs
        word_vectors = svd.fit_transform(term_doc)         # terms x n_comp
        progress.update(1)

        # vectors for the requested target words (lowercased match fallback)
        target_vectors: Dict[str, list] = {}
        if target_words:
            index = {w: i for i, w in enumerate(vocab)}
            for w in target_words:
                i = index.get(w, index.get(w.lower()))
                if i is not None:
                    target_vectors[w] = word_vectors[i].tolist()
        progress.update(1)

        result = {
            "method": "lsa",
            "n_components": n_comp,
            "use_tfidf": use_tfidf,
            "n_documents": len(documents),
            "vocabulary_size": len(vocab),
            "target_vectors": target_vectors,
            "explained_variance_ratio_sum":
                float(svd.explained_variance_ratio_.sum()),
        }

        if save_to is not None:
            # store the full space compactly: vocab + vectors
            payload = dict(result)
            payload["vocabulary"] = vocab
            payload["word_vectors"] = word_vectors.tolist()
            _save_json({"summary": {"stage": "lsa", **result},
                        "data": payload}, save_to)
            progress.update(1)
    if log is not None:
        log.record(
            step=5, operation="estimate_lsa",
            parameters={"n_components": n_comp, "use_tfidf": use_tfidf,
                        "min_df": min_df, "seed": seed},
            justification=justification or
            "Static distributional representation (LSA) over the documents.",
            summary={"n_documents": len(documents),
                     "vocabulary_size": len(vocab),
                     "n_components": n_comp,
                     "explained_variance":
                         result["explained_variance_ratio_sum"]},
            artifact=save_to)
    return result


def documents_from_prepared(prepared: List[Unit],
                            show_progress: bool = False) -> List[str]:
    """Recover one text per document from prepared units.

    LSA works at the document level. Prepared segment units carry context, not
    the full document, so here we group by doc_id and join whatever text each
    document contributes (prepared_text + context). For document-level units
    this is just the document text.
    """
    by_doc: Dict[str, List[str]] = {}
    for u in _progress(prepared, total=len(prepared),
                       desc="Recovering LSA documents", unit="unit",
                       enabled=show_progress):
        piece = u.get("prepared_text", u.get("text", ""))
        left = u.get("left", "")
        right = u.get("right", "")
        by_doc.setdefault(u["doc_id"], []).append(f"{left} {piece} {right}")
    return [" ".join(pieces) for pieces in by_doc.values()]


def _save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
