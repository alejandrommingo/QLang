"""
Phase 4 (part 2): contextual model -- tokenization + offset-to-token alignment.

This is the core of phase 4 for contextual models (BERT, GPT-2, or any other
pretrained model from Hugging Face). It does three things per unit:

  1. tokenize the prepared text with the chosen model's tokenizer (the model
     splits text into sub-word tokens, which do NOT match words: "yellow" may
     become ["yel", "low"]).
  2. ALIGN the segment's character offset with the tokens, so we know exactly
     which tokens are the target segment. This is the article's Figure 2, and
     the whole reason offsets were carried since phase 1.
  3. record everything (tokens, ids, the token indices of the segment), save
     it, and log the decision.

The model is a PARAMETER (default "bert-base-uncased"); change it to "gpt2" or
any other HF model id without touching the code. We only load the TOKENIZER
here (not the heavy model weights), because tokenization + alignment is what
this step produces; running the model to get vectors is phase 5.

Needs: transformers (pip install transformers). The first use downloads the
tokenizer files.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tracelog import TraceLog

Unit = Dict[str, Any]

DEFAULT_MODEL = "bert-base-uncased"


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


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load the tokenizer for a Hugging Face model id.

    Kept in its own function so it is easy to mock/test and so the (possibly
    slow) download happens in one obvious place.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Tokenization needs the 'transformers' library. "
            "Install with: pip install transformers") from e
    return AutoTokenizer.from_pretrained(model_name)


def align_offset_to_tokens(offsets_mapping, segment_offset):
    """Return the token indices that overlap the segment's char offset.

    'offsets_mapping' is the list of (char_start, char_end) per token returned
    by the tokenizer. 'segment_offset' is (start, end) of the segment in the
    SAME text. A token belongs to the segment if its char span overlaps the
    segment span. Special tokens have (0, 0) spans and are skipped.
    """
    seg_start, seg_end = segment_offset
    token_indices = []
    for i, (t_start, t_end) in enumerate(offsets_mapping):
        if t_start == t_end == 0:
            continue  # special token ([CLS], [SEP], etc.)
        # overlap test between [t_start, t_end) and [seg_start, seg_end)
        if t_start < seg_end and seg_start < t_end:
            token_indices.append(i)
    return token_indices


def tokenize_units(units, model_name: str = DEFAULT_MODEL,
                   text_field: str = "prepared_text",
                   tokenizer=None,
                   save_to: Optional[str] = None,
                   log: Optional[TraceLog] = None,
                   justification: str = "",
                   show_progress: bool = False) -> List[Unit]:
    """Tokenize each unit, detecting its type and acting accordingly.

    'units' can be a path to a prepared-sample JSON or an in-memory list.
    'text_field' is which text to tokenize (default 'prepared_text', the
    output of text_prep; falls back to 'text').

    Unit type is detected automatically:
      - SEGMENT (has 'left'/'right' KWIC context): the line "left+segment+
        right" is tokenized and the segment is ALIGNED to its tokens
        ('segment_token_indices'). This is the article's Figure 2.
      - SENTENCE / DOCUMENT (no KWIC context): the whole unit text is
        tokenized; there is no segment to align, so 'segment_token_indices'
        is left empty and 'unit_kind' notes it is the whole unit.

    Each unit gains: tokens, token_ids, segment_token_indices, unit_kind,
    model_name. Saves the output and records the decision in the log.
    """
    if isinstance(units, str):
        units = _load_json_units(units)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    out: List[Unit] = []
    for u in _progress(units, total=len(units),
                       desc="Tokenizing and aligning", unit="unit",
                       enabled=show_progress):
        is_segment = ("left" in u) or ("right" in u)
        segment = u.get(text_field, u.get("text", ""))

        if is_segment:
            left = u.get("left", "")
            right = u.get("right", "")
            line = f"{left}{segment}{right}"
            seg_start = len(left)
            seg_end = len(left) + len(segment)
            kind = "segment"
        else:
            # sentence or document: the whole unit is the observation
            line = segment
            seg_start = seg_end = None
            kind = "sentence_or_document"

        enc = tokenizer(line, return_offsets_mapping=True,
                        add_special_tokens=True)
        offsets_mapping = enc["offset_mapping"]
        token_ids = enc["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        if is_segment:
            seg_token_idx = align_offset_to_tokens(
                offsets_mapping, (seg_start, seg_end))
        else:
            seg_token_idx = []  # no segment to align; whole unit is the target

        new = dict(u)
        new["tokens"] = tokens
        new["token_ids"] = token_ids
        new["segment_token_indices"] = seg_token_idx
        new["unit_kind"] = kind
        new["model_name"] = model_name
        out.append(new)

    if save_to is not None:
        _save_json({"summary": {"stage": "tokenization",
                                "model": model_name, "n_units": len(out)},
                    "data": _serializable(out)}, save_to)
    if log is not None:
        n_segments = sum(1 for u in out if u["unit_kind"] == "segment")
        n_aligned = sum(1 for u in out
                        if u["unit_kind"] == "segment"
                        and u["segment_token_indices"])
        log.record(
            step=4, operation="tokenize_and_align",
            parameters={"model_name": model_name, "text_field": text_field},
            justification=justification or
            "Tokenize prepared text; align segments with their tokens.",
            summary={"n_units": len(out), "model": model_name,
                     "n_segments": n_segments,
                     "n_segments_aligned": n_aligned,
                     "n_whole_units": len(out) - n_segments},
            artifact=save_to)
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_json_units(path: str) -> List[Unit]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    data = content["data"] if isinstance(content, dict) else content
    for u in data:
        if "offset" in u and isinstance(u["offset"], list):
            u["offset"] = tuple(u["offset"])
    return data


def _serializable(units: List[Unit]) -> List[Unit]:
    out = []
    for u in units:
        c = dict(u)
        if "offset" in c and isinstance(c["offset"], tuple):
            c["offset"] = list(c["offset"])
        out.append(c)
    return out


def _save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
