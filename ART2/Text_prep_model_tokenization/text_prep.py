"""
Phase 4 (part 1): model-independent text preparation.

Takes the sample from phase 3 (units with text + offsets) and applies only
MINIMAL, DOCUMENTED transformations, following the article's principle of
minimal intervention: keep the text as produced; change it only with an
explicit, recorded reason.

The delicate part: whenever the text changes, the OFFSET of the segment may
shift. If we collapse two leading spaces, the position of the target word
moves. So this module does not just transform the text -- it RECOMPUTES the
offset so it keeps pointing at the segment inside the transformed text. If the
offset desynchronized, the later offset<->token alignment would silently fail.

Transformations, split by how safe they are:

  SAFE (on by default) -- remove technical noise without losing linguistic
  information:
    - collapse_whitespace : multiple spaces/newlines -> a single space
    - unicode_nfc         : unify equivalent Unicode forms (NFC)

  RISKY (off by default, must be enabled explicitly) -- the article warns these
  can erase informative linguistic variation:
    - lowercase           : "Yellow" -> "yellow" (loses proper-noun casing)
    - strip_accents        : "papá" -> "papa" (loses meaning distinctions)

Every prepared unit records what was done to it. The output is saved as
{summary, data} and logged in the TraceLog, like every other step.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from tracelog import TraceLog

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


# ===========================================================================
# Core idea: transform a string AND map old positions to new positions
# ===========================================================================
# Each transformation returns the new text plus a function that maps any
# character index in the ORIGINAL text to its index in the NEW text. Chaining
# transformations chains these maps, so the segment offset can be carried
# through all of them and stay correct.


def _collapse_whitespace(text: str):
    """Collapse runs of whitespace into a single space.

    Returns (new_text, index_map) where index_map[i] is the position in
    new_text of the character that was at position i in text.
    """
    new_chars: List[str] = []
    index_map: List[int] = [0] * (len(text) + 1)
    prev_was_space = False
    for i, ch in enumerate(text):
        index_map[i] = len(new_chars)
        if ch.isspace():
            if not prev_was_space:
                new_chars.append(" ")
            prev_was_space = True
        else:
            new_chars.append(ch)
            prev_was_space = False
    index_map[len(text)] = len(new_chars)
    return "".join(new_chars), index_map


def _unicode_nfc(text: str):
    """Apply Unicode NFC normalization.

    NFC composes equivalent forms (e.g. 'e' + combining accent -> 'é'). This
    can change string length, so we build a best-effort index map character by
    character. For text that is already NFC (the common case) the map is the
    identity.
    """
    # Fast path: already normalized -> identity map.
    if unicodedata.is_normalized("NFC", text):
        return text, list(range(len(text) + 1))
    # Slow path: normalize per character to keep a usable map.
    new_chars: List[str] = []
    index_map: List[int] = [0] * (len(text) + 1)
    for i, ch in enumerate(text):
        index_map[i] = len(new_chars)
        new_chars.append(unicodedata.normalize("NFC", ch))
    index_map[len(text)] = len("".join(new_chars))
    return "".join(new_chars), index_map


def _lowercase(text: str):
    """Lowercase. RISKY: loses casing information. Length is preserved, so the
    index map is the identity."""
    return text.lower(), list(range(len(text) + 1))


def _strip_accents(text: str):
    """Remove accents/diacritics. RISKY: can change meaning. Done per
    character to preserve length and keep an identity-like map."""
    new_chars: List[str] = []
    index_map: List[int] = [0] * (len(text) + 1)
    for i, ch in enumerate(text):
        index_map[i] = len(new_chars)
        decomposed = unicodedata.normalize("NFD", ch)
        stripped = "".join(c for c in decomposed
                           if unicodedata.category(c) != "Mn")
        new_chars.append(stripped if stripped else ch)
    index_map[len(text)] = len("".join(new_chars))
    return "".join(new_chars), index_map


# registry: name -> (function, is_safe)
_TRANSFORMS = {
    "collapse_whitespace": (_collapse_whitespace, True),
    "unicode_nfc": (_unicode_nfc, True),
    "lowercase": (_lowercase, False),
    "strip_accents": (_strip_accents, False),
}

SAFE_DEFAULTS = ["unicode_nfc", "collapse_whitespace"]


# ===========================================================================
# Apply a chain of transformations to one text, carrying an offset through
# ===========================================================================

def prepare_text(text: str, steps: List[str], offset=None):
    """Apply the named transformations in order to 'text'.

    If 'offset' (start, end) is given, it is remapped through every step so it
    still points at the same segment in the transformed text. Returns
    (new_text, new_offset) where new_offset is None if no offset was given.
    """
    start, end = (offset if offset is not None else (None, None))
    for name in steps:
        if name not in _TRANSFORMS:
            raise ValueError(f"unknown transformation: {name!r}")
        func, _ = _TRANSFORMS[name]
        text, index_map = func(text)
        if offset is not None:
            start = index_map[min(start, len(index_map) - 1)]
            end = index_map[min(end, len(index_map) - 1)]
    new_offset = (start, end) if offset is not None else None
    return text, new_offset


# ===========================================================================
# Prepare a whole sample (the phase-3 units)
# ===========================================================================

def prepare_units(units, steps: Optional[List[str]] = None,
                  save_to: Optional[str] = None,
                  log: Optional[TraceLog] = None,
                  justification: str = "",
                  show_progress: bool = False) -> List[Unit]:
    """Apply text preparation to every unit of a phase-3 sample.

    'units' can be EITHER:
      - a path (str) to the sample JSON saved in phase 3 (e.g. the
        stage2_kwic.json file), which is read automatically; or
      - an in-memory list of units (as returned by the sampling functions).

    'steps' is the ordered list of transformation names. If None, uses the
    safe defaults (unicode_nfc + collapse_whitespace). Risky steps (lowercase,
    strip_accents) must be listed explicitly by the caller.

    Each prepared unit keeps its original text/offset and adds:
      - prepared_text   : the transformed text of the unit
      - transformations : the list of steps applied
    Context (left/right) is transformed too. Saves the output and records the
    decision in the log.
    """
    # accept a file path or an already-loaded list
    if isinstance(units, str):
        units = _load_sample(units)

    steps = list(SAFE_DEFAULTS if steps is None else steps)
    risky = [s for s in steps if not _TRANSFORMS.get(s, (None, True))[1]]

    prepared: List[Unit] = []
    for u in _progress(units, total=len(units), desc="Preparing text",
                       unit="unit", enabled=show_progress):
        new = dict(u)
        new_text, _ = prepare_text(u["text"], steps)
        new["prepared_text"] = new_text
        new["transformations"] = steps
        if "left" in u:
            new["left"], _ = prepare_text(u["left"], steps)
        if "right" in u:
            new["right"], _ = prepare_text(u["right"], steps)
        prepared.append(new)

    if save_to is not None:
        _save_units(prepared, save_to, stage="text_prep")
    if log is not None:
        log.record(
            step=4, operation="prepare_text",
            parameters={"steps": steps, "risky_steps_used": risky},
            justification=justification or
            "Minimal text preparation before tokenization.",
            summary={"n_units": len(prepared),
                     "transformations": steps,
                     "risky_used": bool(risky)},
            artifact=save_to)
    return prepared


# ===========================================================================
# Persistence (same {summary, data} shape as the rest of the pipeline)
# ===========================================================================

def _load_sample(path: str) -> List[Unit]:
    """Read a phase-3 sample JSON ({summary, data} or a bare list).

    Restores offsets as tuples, like sampling.load_units does, so text_prep
    can be fed directly with the file the sampling step saved.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    data = content["data"] if isinstance(content, dict) else content
    for u in data:
        if "offset" in u and isinstance(u["offset"], list):
            u["offset"] = tuple(u["offset"])
    return data


def _save_units(units: List[Unit], path: str, stage: str = "") -> None:
    serializable = []
    for u in units:
        copy = dict(u)
        if "offset" in copy and isinstance(copy["offset"], tuple):
            copy["offset"] = list(copy["offset"])
        serializable.append(copy)
    wrapper = {
        "summary": {"stage": stage, "n_units": len(units)},
        "data": serializable,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)


def load_prepared(path: str) -> List[Unit]:
    """Reload prepared units saved by prepare_units."""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    data = content["data"] if isinstance(content, dict) else content
    for u in data:
        if "offset" in u and isinstance(u["offset"], list):
            u["offset"] = tuple(u["offset"])
    return data
