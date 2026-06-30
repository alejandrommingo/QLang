"""
Aggregation: composing one vector per unit from its token vectors.

The model gives one vector per token, but the unit of analysis (a target word,
a sentence, a document) usually spans several tokens. This module recomposes
the unit's vector from its tokens, using the alignment computed earlier.

Two strategies, as the methodology describes:

  - MEAN POOLING: average the token vectors. Integrates information spread
    across tokens, though it can dilute signals concentrated in one position.
  - REPRESENTATIVE TOKEN: instead of averaging, take a single token's vector
    to stand for the whole set. For a word split into several sub-tokens (e.g.
    "paraguas" -> "par","aguas"), this means taking just one of them: the
    first ([CLS]-style, for bidirectional models like BERT) or the last (for
    autoregressive models like GPT). Only appropriate when the model is trained
    so that one token summarizes the sequence; otherwise that single token
    doesn't really represent the set, and mean pooling is safer.

Which tokens are aggregated depends on the unit type:
  - SEGMENT: aggregate only the segment's tokens (segment_token_indices) -- the
    representation of the target word in its context.
  - SENTENCE / DOCUMENT: aggregate all the unit's tokens (the whole text is the
    observation).

Input: units with 'token_vectors'. Output: each unit gains 'vector' (its final
representation). Uses numpy.
"""

from __future__ import annotations

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


MEAN = "mean_pooling"
REPRESENTATIVE = "representative_token"
STRATEGIES = (MEAN, REPRESENTATIVE)


def aggregate_units(units: List[Unit], strategy: str = MEAN,
                    representative: str = "cls",
                    save_to: Optional[str] = None,
                    log=None, justification: str = "",
                    show_progress: bool = False) -> List[Unit]:
    """Compose one vector per unit from its token vectors.

    'strategy' is 'mean_pooling' or 'representative_token'. For the
    representative strategy, 'representative' is 'cls' (first token, for
    bidirectional models) or 'last' (last token, for autoregressive models).

    For segment units, only the segment's tokens are used; for sentence/
    document units, all tokens are used. Each unit gains 'vector' and
    'aggregation'. Units whose segment has no aligned tokens get vector None
    (and are flagged), rather than guessing. If 'save_to' is given, saves one
    vector per unit (reasonable size) without the heavy per-token vectors.
    """
    import numpy as np

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    out: List[Unit] = []
    n_ok = 0
    for u in _progress(units, total=len(units), desc="Aggregating vectors",
                       unit="unit", enabled=show_progress):
        token_vecs = u.get("token_vectors")
        if not token_vecs:
            v = dict(u)
            v["vector"] = None
            v["aggregation"] = "skipped_no_vectors"
            out.append(v)
            continue

        vecs = np.array(token_vecs, dtype=float)
        kind = u.get("unit_kind", "segment")

        # which token positions to aggregate
        if kind == "segment":
            idx = u.get("segment_token_indices") or []
        else:
            # whole-text unit: all real tokens
            idx = list(range(len(token_vecs)))

        if not idx:
            v = dict(u)
            v["vector"] = None
            v["aggregation"] = "skipped_no_aligned_tokens"
            out.append(v)
            continue

        if strategy == MEAN:
            vector = vecs[idx].mean(axis=0)
        else:  # REPRESENTATIVE
            pos = idx[0] if representative == "cls" else idx[-1]
            vector = vecs[pos]

        v = dict(u)
        v["vector"] = vector.tolist()
        v["aggregation"] = strategy
        if strategy == REPRESENTATIVE:
            v["representative"] = representative
        out.append(v)
        n_ok += 1

    if save_to is not None:
        _save_aggregated(out, save_to, strategy)
    if log is not None:
        log.record(
            step=5, operation="aggregate",
            parameters={"strategy": strategy,
                        "representative": representative
                        if strategy == REPRESENTATIVE else None},
            justification=justification or
            "Compose one vector per unit from its token vectors.",
            summary={"n_units": len(out), "n_with_vector": n_ok,
                     "strategy": strategy},
            artifact=save_to)
    return out


def _save_aggregated(units: List[Unit], path: str, strategy: str) -> None:
    """Save one vector per unit, dropping the heavy per-token vectors."""
    import json
    light = []
    for u in units:
        c = {k: v for k, v in u.items() if k != "token_vectors"}
        light.append(c)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"stage": "aggregation", "strategy": strategy,
                               "n_units": len(units)},
                   "data": light}, f, ensure_ascii=False, indent=2)
