"""
Fragmentation: handling texts longer than the model's maximum input length.

Contextual models accept a limited number of tokens (e.g. 512 for BERT). When
a tokenized unit exceeds that limit, it must be fragmented. This module
implements the strategies the methodology describes, and adapts them to the
unit type:

  SEGMENT units (a target word with its KWIC context):
    The observation is the word in context, and the representation is taken
    from the word's tokens. So we must NEVER lose or cut the segment. Instead
    of fragmenting blindly, we take a single window CENTERED on the segment:
    the segment plus as much context as fits on each side, up to the limit.
    This guarantees the target word is always present and surrounded by
    context.

  SENTENCE / DOCUMENT units (the whole text is the observation):
    Here we want to process the whole text, so we use the generic strategies:
      - truncate       : keep the first max_tokens, drop the rest (loses info)
      - fixed windows  : split into non-overlapping blocks
      - sliding windows: overlapping blocks, parameterized by window size and
                         stride (e.g. window 512 / stride 256 = 50% overlap;
                         stride == window = fixed windows with no overlap)

Each fragmented unit records how it was fragmented, so the decision is
traceable. Batching/padding for running the model is handled later, when the
model is actually executed.
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


# fragmentation strategies for whole-text units
TRUNCATE = "truncate"
FIXED = "fixed_windows"
SLIDING = "sliding_windows"
STRATEGIES = (TRUNCATE, FIXED, SLIDING)


def fragment_units(units: List[Unit], max_tokens: int = 512,
                   strategy: str = SLIDING, window: Optional[int] = None,
                   stride: Optional[int] = None,
                   special_tokens: int = 2,
                   save_to: Optional[str] = None,
                   show_progress: bool = False) -> List[Unit]:
    """Fragment the units that exceed the model's token limit.

    'max_tokens' is the model's maximum input length (including special
    tokens). 'special_tokens' is how many slots are reserved for them (2 for
    BERT's [CLS]/[SEP]); the usable budget is max_tokens - special_tokens.

    Segment units get a single window centered on the segment. Sentence/
    document units use 'strategy' (truncate / fixed / sliding). For sliding,
    'window' and 'stride' default to max_tokens and max_tokens//2.

    Returns a new list. A unit short enough is returned unchanged (with
    n_fragments = 1). A long unit may become several fragment-units, each
    carrying 'fragment_index', 'n_fragments', and the token slice it covers.
    'save_to' is optional.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    budget = max_tokens - special_tokens
    window = window or max_tokens
    stride = stride or (max_tokens // 2)

    out: List[Unit] = []
    for u in _progress(units, total=len(units), desc="Fragmenting units",
                       unit="unit", enabled=show_progress):
        n_tokens = len(u.get("tokens", []))
        # short enough: no fragmentation needed
        if n_tokens <= max_tokens:
            v = dict(u)
            v["fragment_index"] = 0
            v["n_fragments"] = 1
            v["fragmentation"] = "none"
            out.append(v)
            continue

        kind = u.get("unit_kind", "segment")
        if kind == "segment":
            out.append(_center_on_segment(u, budget))
        else:
            out.extend(_fragment_whole(u, strategy, window, stride,
                                       special_tokens))

    if save_to is not None:
        import json
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump({"summary": {"stage": "fragmentation",
                                   "strategy": strategy, "n_units": len(out)},
                       "data": out}, f, ensure_ascii=False, indent=2)
    return out


def _center_on_segment(unit: Unit, budget: int) -> Unit:
    """Take one window centered on the segment, so the target word is kept
    with as much context as fits on each side."""
    idx = unit.get("segment_token_indices") or []
    tokens = unit["tokens"]
    if not idx:
        # no aligned segment: fall back to keeping the start
        seg_lo, seg_hi = 0, min(len(tokens), budget)
    else:
        seg_lo, seg_hi = min(idx), max(idx) + 1

    seg_len = seg_hi - seg_lo
    spare = max(budget - seg_len, 0)
    left_room = spare // 2
    right_room = spare - left_room

    start = max(0, seg_lo - left_room)
    end = min(len(tokens), seg_hi + right_room)
    # if we hit one edge, use the leftover on the other side
    if start == 0:
        end = min(len(tokens), start + budget)
    if end == len(tokens):
        start = max(0, end - budget)

    v = dict(unit)
    v["tokens"] = tokens[start:end]
    if "token_ids" in unit:
        v["token_ids"] = unit["token_ids"][start:end]
    # shift the segment indices into the new window
    v["segment_token_indices"] = [i - start for i in idx
                                  if start <= i < end]
    v["fragment_index"] = 0
    v["n_fragments"] = 1
    v["fragmentation"] = "centered_on_segment"
    v["fragment_token_range"] = [start, end]
    return v


def _fragment_whole(unit: Unit, strategy: str, window: int, stride: int,
                    special_tokens: int) -> List[Unit]:
    """Fragment a whole-text unit (sentence/document) by the chosen strategy."""
    tokens = unit["tokens"]
    ids = unit.get("token_ids", [])
    n = len(tokens)
    budget = window - special_tokens

    if strategy == TRUNCATE:
        ranges = [(0, min(n, budget))]
    elif strategy == FIXED:
        ranges = [(s, min(n, s + budget)) for s in range(0, n, budget)]
    else:  # SLIDING
        step = max(stride - special_tokens, 1)
        ranges = []
        s = 0
        while s < n:
            ranges.append((s, min(n, s + budget)))
            if s + budget >= n:
                break
            s += step

    frags: List[Unit] = []
    for fi, (a, b) in enumerate(ranges):
        v = dict(unit)
        v["tokens"] = tokens[a:b]
        if ids:
            v["token_ids"] = ids[a:b]
        v["segment_token_indices"] = []  # whole-text units have no segment
        v["fragment_index"] = fi
        v["n_fragments"] = len(ranges)
        v["fragmentation"] = strategy
        v["fragment_token_range"] = [a, b]
        frags.append(v)
    return frags
