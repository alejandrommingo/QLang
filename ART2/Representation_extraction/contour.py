"""
Contour: the occurrences x dimensions matrix for a segment.

When the unit is a segment, the point is usually not a single isolated vector
but the term's behaviour ACROSS its contexts. The contour assembles every
occurrence's vector into one matrix:

    rows    = occurrences of the target word (one per context)
    columns = dimensions of the representation

This matrix is the term's semantic portrait: it is what later lets you compare
contexts, measure the spread of senses, or compare the term between conditions
(the comparative scale). Each row keeps its source reference (article, context)
so rows can be grouped afterwards.

For sentence/document units there is no single target word repeated across
contexts, so a contour in this sense does not apply; those units are left as a
plain list of vectors.

Input: units with 'vector' (from aggregation). Output: a contour dict with the
matrix and the per-row metadata, saved to JSON. Uses numpy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

Unit = Dict[str, Any]


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


def build_contour(units: List[Unit], word: Optional[str] = None,
                  save_to: Optional[str] = None,
                  log=None, justification: str = "",
                  show_progress: bool = False) -> Dict[str, Any]:
    """Assemble the occurrences x dimensions matrix from segment units.

    Only units that have a vector are included (those whose segment could be
    aligned and aggregated). Each row keeps metadata: the source document, the
    context, and any reference, so rows can be grouped later (by article, by
    condition, …).

    If the same occurrence appears in more than one unit (for example because
    a long text was split into overlapping sliding windows), it is counted
    ONCE: rows are deduplicated by (doc_id, segment offset). This mirrors the
    overlap handling needed when windows overlap.

    Returns a dict with:
      - matrix    : list of rows (each a vector) -> occurrences x dimensions
      - rows_meta : per-row metadata, aligned with 'matrix'
      - shape     : [n_occurrences, n_dimensions]
    Saves it and records the decision.
    """
    import numpy as np

    # Group rows by occurrence identity. The same occurrence (same word
    # position in the same document) can appear in several units when a long
    # text was split into overlapping windows. Instead of keeping just the
    # first, we MEAN-POOL all its versions, which is more robust.
    groups = {}        # key -> {"vectors": [...], "meta": {...}}
    order = []         # preserve first-seen order
    for u in _progress(units, total=len(units), desc="Building contour",
                       unit="unit", enabled=show_progress):
        vec = u.get("vector")
        if vec is None:
            continue
        offset = u.get("offset")
        key = (u.get("doc_id"),
               tuple(offset) if isinstance(offset, (list, tuple)) else offset)
        # rows with no identity (no doc/offset) are each kept on their own
        if key == (None, None):
            key = ("__unkeyed__", len(order))
        if key not in groups:
            groups[key] = {"vectors": [], "meta": {
                "doc_id": u.get("doc_id"),
                "text": u.get("text"),
                "left": u.get("left", ""),
                "right": u.get("right", ""),
                "reference": u.get("reference"),
            }}
            order.append(key)
        groups[key]["vectors"].append(vec)

    n_occurrences_raw = sum(len(g["vectors"]) for g in groups.values())
    n_duplicates = n_occurrences_raw - len(order)

    rows = []
    rows_meta = []
    for key in order:
        g = groups[key]
        pooled = np.array(g["vectors"], dtype=float).mean(axis=0)
        rows.append(pooled.tolist())
        meta = dict(g["meta"])
        meta["n_pooled"] = len(g["vectors"])  # how many windows were averaged
        rows_meta.append(meta)

    if not rows:
        raise ValueError(
            "No occurrences with a vector to build a contour. Check that the "
            "segments were aligned and aggregated.")

    matrix = np.array(rows, dtype=float)
    contour = {
        "word": word,
        "matrix": matrix.tolist(),
        "rows_meta": rows_meta,
        "shape": list(matrix.shape),
    }

    if save_to is not None:
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump({"summary": {"stage": "contour", "word": word,
                                   "shape": list(matrix.shape),
                                   "duplicates_removed": n_duplicates},
                       "data": contour}, f, ensure_ascii=False, indent=2)
    if log is not None:
        log.record(
            step=5, operation="build_contour",
            parameters={"word": word},
            justification=justification or
            "Assemble the occurrences x dimensions matrix (the term's "
            "contour).",
            summary={"n_occurrences": int(matrix.shape[0]),
                     "n_dimensions": int(matrix.shape[1]),
                     "duplicates_removed": n_duplicates},
            artifact=save_to)
    return contour
