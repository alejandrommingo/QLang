"""
Step 3 of the methodological pathway (sampling).

Brings together the two dimensions the methodology distinguishes:

  A) TARGET UNIT (extractors) -- what the observation is:
       extract_segments / extract_sentences / extract_documents
     Each extractor turns a corpus (step 1) into a list of "units" in the
     common format, always carrying doc_id, text, offset and meta.

  B) SAMPLING STRATEGY (generic) -- how cases are chosen:
       exhaustive / random_simple / stratified / reservoir
     They operate on any list of units without knowing what is inside.

Plus cross-cutting controls (cap per group, summary) and two-stage segment
sampling that produces a KWIC concordance:

  STAGE 1 (stage1_documents): documents containing the segment, reduced to N.
  STAGE 2 (stage2_occurrences): occurrences of the segment with offsets, a
    per-document cap, a strategy, and the context window (KWIC).

Auditing utilities (save_units, the TraceLog) keep every step traceable.
It relies on corpus_loading for the common format, so step 1 and step 3 speak
exactly the same data language. It never alters the text (minimal
intervention): offsets point to the original body.
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import corpus_loading as cl
from tracelog import TraceLog

Corpus = cl.Corpus
Unit = Dict[str, Any]
Key = Callable[[Unit], Any]


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
# A) TARGET UNIT -- extractors
# ===========================================================================

MATCH_EXACT = "exact"        # "yellow" yes; "yellows", "yellowish" no
MATCH_VARIANTS = "variants"  # "yellow", "yellows"...
MATCH_LOOSE = "loose"        # any appearance, even inside another word
MATCH_MODES = (MATCH_EXACT, MATCH_VARIANTS, MATCH_LOOSE)


def build_pattern(word: str, mode: str = MATCH_EXACT) -> re.Pattern:
    """Build the regular expression for the word, by mode. Case-insensitive."""
    if mode not in MATCH_MODES:
        raise ValueError(f"mode '{mode}' not valid; use {MATCH_MODES}")
    root = re.escape(word)
    if mode == MATCH_EXACT:
        pattern = rf"\b{root}\b"
    elif mode == MATCH_VARIANTS:
        pattern = rf"\b{root}[a-záéíóúñ]{{0,3}}\b"
    else:  # MATCH_LOOSE
        pattern = root
    return re.compile(pattern, re.IGNORECASE)


def _unit(doc_id: str, text: str, start: int, end: int,
          meta: Dict) -> Unit:
    """Build a unit in the common format."""
    return {"doc_id": doc_id, "text": text,
            "offset": (start, end), "meta": dict(meta)}


def extract_segments(corpus: Corpus, word: str,
                     mode: str = MATCH_EXACT,
                     show_progress: bool = False) -> List[Unit]:
    """Each appearance of 'word' in the corpus, as a unit with its offset.

    This is the second stage of the two-stage segment sampling: occurrences
    are located within the documents (document selection is done by the
    strategy, or by an external filter).
    """
    cl.validate_corpus(corpus)
    pattern = build_pattern(word, mode)
    units: List[Unit] = []
    items = corpus.items()
    for doc_id, doc in _progress(items, total=len(corpus),
                                 desc="Finding segment occurrences",
                                 unit="doc", enabled=show_progress):
        body = doc["body"]
        meta = doc.get("meta", {})
        for m in pattern.finditer(body):
            units.append(
                _unit(doc_id, body[m.start():m.end()],
                      m.start(), m.end(), meta))
    return units


def extract_sentences(corpus: Corpus,
                      show_progress: bool = False) -> List[Unit]:
    """One unit per sentence. Splits on . ! ? followed by whitespace.

    The methodology warns sentence splitting is delicate (abbreviations,
    etc.); this criterion is simple and explicit, improvable in step 4.
    """
    cl.validate_corpus(corpus)
    units: List[Unit] = []
    items = corpus.items()
    for doc_id, doc in _progress(items, total=len(corpus),
                                 desc="Extracting sentences", unit="doc",
                                 enabled=show_progress):
        body = doc["body"]
        meta = doc.get("meta", {})
        for start, end in _sentence_bounds(body):
            raw = body[start:end]
            text = raw.strip()
            if not text:
                continue
            shift = raw.index(text)
            real_start = start + shift
            units.append(
                _unit(doc_id, text, real_start, real_start + len(text), meta))
    return units


def _sentence_bounds(body: str) -> List[tuple]:
    cuts = [m.end() for m in re.finditer(r"[.!?]+\s+", body)]
    bounds, prev = [], 0
    for c in cuts:
        bounds.append((prev, c))
        prev = c
    if prev < len(body):
        bounds.append((prev, len(body)))
    return bounds


def extract_documents(corpus: Corpus,
                      show_progress: bool = False) -> List[Unit]:
    """Each whole document as a unit (the unit is its own source)."""
    cl.validate_corpus(corpus)
    units: List[Unit] = []
    items = corpus.items()
    for doc_id, doc in _progress(items, total=len(corpus),
                                 desc="Extracting documents", unit="doc",
                                 enabled=show_progress):
        body = doc["body"]
        units.append(
            _unit(doc_id, body, 0, len(body), doc.get("meta", {})))
    return units


def select_documents_with_word(corpus: Corpus, word: str,
                               mode: str = MATCH_EXACT,
                               show_progress: bool = False) -> List[Unit]:
    """Documents containing the word, as document-units.

    This is the base of STAGE 1 of segment sampling: out of the whole corpus
    we keep only the documents where the segment appears, before applying a
    strategy to reduce their number.
    """
    cl.validate_corpus(corpus)
    pattern = build_pattern(word, mode)
    items = corpus.items()
    return [_unit(doc_id, doc["body"], 0, len(doc["body"]),
                  doc.get("meta", {}))
            for doc_id, doc in _progress(
                items, total=len(corpus), desc="Selecting documents",
                unit="doc", enabled=show_progress)
            if pattern.search(doc["body"])]


# ===========================================================================
# B) SAMPLING STRATEGY -- generic (work for any unit)
# ===========================================================================

def exhaustive(units: List[Unit]) -> List[Unit]:
    """All units (a copy). Avoids selection bias when the volume is small."""
    return list(units)


def random_simple(units: List[Unit], n: int,
                  seed: Optional[int] = None) -> List[Unit]:
    """Simple random sampling: n units at random, all equally likely.

    The most basic case: no strata, no groups, just "give me n at random from
    this list". Useful when the set fits in memory and there is no condition
    to preserve. If n exceeds the total, returns everything. Reproducible
    with 'seed'.
    """
    if n >= len(units):
        return list(units)
    rng = random.Random(seed)
    return rng.sample(units, n)


def stratified(units: List[Unit], n_total: int, key: Key,
               seed: Optional[int] = None,
               show_progress: bool = False) -> List[Unit]:
    """n_total units preserving each stratum's proportion.

    Groups units by 'key(u)' and gives each group a quota proportional to its
    size. The 'key' is chosen by the caller (e.g. the source or the
    language), so the strategy does not depend on the unit's internal shape.
    """
    if n_total >= len(units):
        return list(units)
    rng = random.Random(seed)

    groups: Dict[Any, List[Unit]] = {}
    for u in _progress(units, total=len(units),
                       desc="Grouping strata", unit="unit",
                       enabled=show_progress):
        groups.setdefault(key(u), []).append(u)

    total = len(units)
    result: List[Unit] = []
    assigned = 0
    grouped = groups.values()
    for group in _progress(grouped, total=len(groups),
                           desc="Sampling strata", unit="stratum",
                           enabled=show_progress):
        quota = min(int(math.floor(n_total * len(group) / total)), len(group))
        result.extend(rng.sample(group, quota))
        assigned += quota

    missing = n_total - assigned
    if missing > 0:
        chosen = set(id(u) for u in result)
        remaining = [u for u in units if id(u) not in chosen]
        rng.shuffle(remaining)
        result.extend(remaining[:missing])
    return result


def reservoir(stream: Iterable[Unit], n: int,
              seed: Optional[int] = None,
              show_progress: bool = False) -> List[Unit]:
    """Fixed-size random sample of size n over a stream (algorithm R, Vitter).

    Walks the stream once, without knowing the total beforehand. For massive
    or sequentially processed corpora.
    """
    rng = random.Random(seed)
    pool: List[Unit] = []
    for i, elem in enumerate(_progress(stream, desc="Reservoir sampling",
                                       unit="unit",
                                       enabled=show_progress)):
        if i < n:
            pool.append(elem)
        else:
            j = rng.randint(0, i)
            if j < n:
                pool[j] = elem
    return pool


# ===========================================================================
# Cross-cutting controls
# ===========================================================================

def cap_per_group(units: List[Unit], max_per_group: int, key: Key,
                  seed: Optional[int] = None,
                  show_progress: bool = False) -> List[Unit]:
    """Cap how many units each group contributes (e.g. each document), so a
    single source does not dominate the sample."""
    rng = random.Random(seed)
    groups: Dict[Any, List[Unit]] = {}
    for u in _progress(units, total=len(units),
                       desc="Grouping for cap", unit="unit",
                       enabled=show_progress):
        groups.setdefault(key(u), []).append(u)
    result: List[Unit] = []
    grouped = groups.values()
    for group in _progress(grouped, total=len(groups),
                           desc="Applying group cap", unit="group",
                           enabled=show_progress):
        result.extend(rng.sample(group, max_per_group)
                      if len(group) > max_per_group else group)
    return result


def summary_per_group(units: List[Unit], key: Key) -> Dict[Any, int]:
    """How many units each group contributes. To review the spread."""
    counts: Dict[Any, int] = {}
    for u in units:
        k = key(u)
        counts[k] = counts.get(k, 0) + 1
    return counts


def cap_per_group_spaced(units: List[Unit], max_per_group: int, key: Key,
                         min_distance: int = 500,
                         seed: Optional[int] = None,
                         show_progress: bool = False) -> List[Unit]:
    """Cap units per group, at random but keeping them spaced apart.

    Within each group it picks up to 'max_per_group' occurrences at random,
    preferring those at least 'min_distance' characters apart (measured on the
    offset), so the chosen occurrences come from diverse contexts rather than
    clustering together.

    If the full distance does not yield enough occurrences (e.g. a short
    article), the distance is RELAXED PROGRESSIVELY (halved, then halved
    again, down to 0) and the selection is topped up, so the article still
    contributes up to 'max_per_group' occurrences -- as spaced as the text
    allows. Better to take 3 somewhat-closer occurrences than to be left with
    just 1.

    The selection stays random within each distance level. Only meaningful
    for units that have an offset within the same document (segments).
    """
    rng = random.Random(seed)
    groups: Dict[Any, List[Unit]] = {}
    for u in _progress(units, total=len(units),
                       desc="Grouping for spaced cap", unit="unit",
                       enabled=show_progress):
        groups.setdefault(key(u), []).append(u)

    result: List[Unit] = []
    grouped = groups.values()
    for group in _progress(grouped, total=len(groups),
                           desc="Applying spaced cap", unit="group",
                           enabled=show_progress):
        if len(group) <= max_per_group:
            result.extend(group)
            continue
        result.extend(_pick_spaced(group, max_per_group, min_distance, rng))
    return result


def _pick_spaced(group: List[Unit], target: int, min_distance: int,
                 rng: random.Random) -> List[Unit]:
    """Pick up to 'target' occurrences from one group, as spaced as possible.

    Tries the full distance first; if it cannot reach 'target', halves the
    distance and tops up, repeating down to distance 0. Already-chosen
    occurrences are kept across levels.
    """
    accepted: List[Unit] = []
    distance = min_distance
    while True:
        shuffled = list(group)
        rng.shuffle(shuffled)
        for u in shuffled:
            if len(accepted) >= target:
                break
            if u in accepted:
                continue
            pos = u["offset"][0]
            if all(abs(pos - a["offset"][0]) >= distance for a in accepted):
                accepted.append(u)
        if len(accepted) >= target or distance == 0:
            break
        distance = distance // 2 if distance > 1 else 0  # relax progressively
    return accepted


# ===========================================================================
# Context window (KWIC) -- the "relevant text around the segment"
# ===========================================================================
# Conceptually this belongs to step 4 (how much context accompanies the
# unit), but it is computed from the offset, so it fits right after sampling.
# The unit is STILL the segment; the window is only its surroundings.

WINDOW_CHARS = "chars"
WINDOW_WORDS = "words"
WINDOW_SENTENCE = "sentence"
WINDOW_PARAGRAPH = "paragraph"
WINDOW_MODES = (WINDOW_CHARS, WINDOW_WORDS, WINDOW_SENTENCE, WINDOW_PARAGRAPH)


def extract_window(corpus: Corpus, occurrence: Unit,
                   mode: str = WINDOW_CHARS, size: int = 50) -> Unit:
    """Return a copy of the occurrence with its context window added.

    'mode' decides how it is measured:
      - 'chars'    : 'size' characters on each side.
      - 'words'    : 'size' words on each side.
      - 'sentence' : the whole sentence containing the segment.
      - 'paragraph': the whole paragraph containing the segment.

    Adds three keys: 'left', 'right' (context on each side, as in a KWIC
    concordance) and 'window_offset'. The segment itself stays in 'text', so
    a KWIC line is left + text + right.
    """
    if mode not in WINDOW_MODES:
        raise ValueError(f"window mode '{mode}' not valid; use {WINDOW_MODES}")
    body = corpus[occurrence["doc_id"]]["body"]
    start, end = occurrence["offset"]

    if mode == WINDOW_CHARS:
        w_start = max(0, start - size)
        w_end = min(len(body), end + size)
    elif mode == WINDOW_WORDS:
        w_start, w_end = _window_by_words(body, start, end, size)
    elif mode == WINDOW_SENTENCE:
        w_start, w_end = _window_by_sentence(body, start, end)
    else:  # WINDOW_PARAGRAPH
        w_start, w_end = _window_by_paragraph(body, start, end)

    new = dict(occurrence)
    new["left"] = body[w_start:start]
    new["right"] = body[end:w_end]
    new["window_offset"] = (w_start, w_end)
    return new


def _window_by_words(body: str, start: int, end: int, n_words: int) -> tuple:
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", body)]
    idx = next((i for i, (s, e) in enumerate(words) if s <= start < e), None)
    if idx is None:
        return (start, end)
    i0 = max(0, idx - n_words)
    i1 = min(len(words) - 1, idx + n_words)
    return (words[i0][0], words[i1][1])


def _window_by_sentence(body: str, start: int, end: int) -> tuple:
    starts = [0] + [m.end() for m in re.finditer(r"[.!?]+\s+", body)]
    ends = [m.end() for m in re.finditer(r"[.!?]+", body)] + [len(body)]
    s_start = max((s for s in starts if s <= start), default=0)
    s_end = min((e for e in ends if e >= end), default=len(body))
    return (s_start, s_end)


def _window_by_paragraph(body: str, start: int, end: int) -> tuple:
    """Offsets of the paragraph containing the occurrence.

    A paragraph is a block separated by one or more blank lines. In texts
    without line breaks the paragraph is the whole body.
    """
    cuts = [m.start() for m in re.finditer(r"\n\s*\n", body)]
    ends = [m.end() for m in re.finditer(r"\n\s*\n", body)]
    p_start = max([0] + [e for e in ends if e <= start])
    after = [c for c in cuts if c >= end]
    p_end = min(after) if after else len(body)
    return (p_start, p_end)


# ===========================================================================
# Auditing: persistence of results
# ===========================================================================

def save_units(units: List[Unit], path: str, stage: str = "") -> None:
    """Save a list of units to JSON, with a count summary at the top.

    File shape:
        {"summary": {n_units, n_articles, ...}, "data": [...]}
    so opening it shows how many elements there are at a glance. Offsets
    (tuples) are stored as lists because JSON has no tuples.
    """
    serializable = []
    for u in units:
        copy = dict(u)
        copy["offset"] = list(u["offset"])
        serializable.append(copy)

    summary = {
        "stage": stage,
        "n_units": len(units),
        "n_articles": len({u["doc_id"] for u in units}),
    }
    wrapper = {"summary": summary, "data": serializable}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)


def load_units(path: str) -> List[Unit]:
    """Reload saved units, restoring offsets as tuples.

    Accepts both the new format ({summary, data}) and the old one (a bare
    list), so files saved with previous versions are not broken.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    data = content["data"] if isinstance(content, dict) else content
    for u in data:
        u["offset"] = tuple(u["offset"])
    return data


# ===========================================================================
# Single-stage orchestrator
# ===========================================================================

def sample(corpus: Corpus, unit: str, strategy: str,
           log: Optional[TraceLog] = None,
           justification: str = "",
           artifact: Optional[str] = None,
           word: Optional[str] = None,
           mode: str = MATCH_EXACT,
           n_total: Optional[int] = None,
           key: Optional[Key] = None,
           max_per_group: Optional[int] = None,
           group_key: Optional[Key] = None,
           seed: Optional[int] = None,
           show_progress: bool = False) -> List[Unit]:
    """Orchestrate one extraction + one strategy, save and leave a trace.

    Combines the unit dimension ('segment'|'sentence'|'document') with the
    strategy dimension ('exhaustive'|'random'|'stratified'|'reservoir'). If
    'log' is given, records the decision. If 'artifact' is given, saves the
    result so it can be audited.
    """
    if unit == "segment":
        if word is None:
            raise ValueError("unit 'segment' requires 'word'")
        units = extract_segments(corpus, word, mode,
                                 show_progress=show_progress)
    elif unit == "sentence":
        units = extract_sentences(corpus, show_progress=show_progress)
    elif unit == "document":
        units = extract_documents(corpus, show_progress=show_progress)
    else:
        raise ValueError(f"unit '{unit}' not valid")

    if max_per_group is not None:
        k = group_key or (lambda u: u["doc_id"])
        units = cap_per_group(units, max_per_group, k, seed,
                              show_progress=show_progress)

    selection = _apply_strategy(units, strategy, n_total, key, seed,
                                show_progress=show_progress)

    if artifact is not None:
        save_units(selection, artifact, stage=f"sample[{unit}/{strategy}]")
    if log is not None:
        log.record(
            step=3,
            operation=f"sample[{unit}/{strategy}]",
            parameters={"unit": unit, "strategy": strategy, "word": word,
                        "mode": mode, "n_total": n_total,
                        "max_per_group": max_per_group, "seed": seed},
            justification=justification,
            summary={"n_units": len(selection),
                     "n_articles": len({u["doc_id"] for u in selection})},
            artifact=artifact,
        )
    return selection


# ===========================================================================
# TWO-STAGE segment sampling (KWIC concordance)
# ===========================================================================

def _apply_strategy(units: List[Unit], strategy: str,
                    n_total: Optional[int], key: Optional[Key],
                    seed: Optional[int],
                    show_progress: bool = False) -> List[Unit]:
    """Apply the generic strategy named by 'strategy'."""
    if strategy == "exhaustive":
        return exhaustive(units)
    if strategy == "random":
        if n_total is None:
            raise ValueError("'random' requires n_total")
        return random_simple(units, n_total, seed)
    if strategy == "stratified":
        if n_total is None or key is None:
            raise ValueError("'stratified' requires n_total and key")
        return stratified(units, n_total, key, seed,
                          show_progress=show_progress)
    if strategy == "reservoir":
        if n_total is None:
            raise ValueError("'reservoir' requires n_total")
        return reservoir(iter(units), n_total, seed,
                         show_progress=show_progress)
    raise ValueError(f"strategy '{strategy}' not valid")


def stage1_documents(corpus: Corpus, word: str,
                     strategy: str = "exhaustive",
                     n_documents: Optional[int] = None,
                     key: Optional[Key] = None,
                     mode: str = MATCH_EXACT,
                     seed: Optional[int] = None,
                     save_to: Optional[str] = None,
                     log: Optional[TraceLog] = None,
                     justification: str = "",
                     show_progress: bool = False) -> Corpus:
    """Select the documents containing the segment and reduce them to N.

    Returns a sub-corpus (step 1 format) and, if 'save_to' is given, persists
    it so stage 2 can reuse it. Records the decision in the log.
    """
    candidates = select_documents_with_word(
        corpus, word, mode, show_progress=show_progress)
    chosen = _apply_strategy(candidates, strategy, n_documents, key, seed,
                             show_progress=show_progress)
    subcorpus: Corpus = {
        u["doc_id"]: cl.make_document(u["text"], u["meta"]) for u in chosen
    }
    if save_to is not None:
        cl.save_corpus(subcorpus, save_to, stage="stage1_documents")
    if log is not None:
        log.record(
            step=3, operation=f"stage1_documents[{strategy}]",
            parameters={"word": word, "mode": mode, "strategy": strategy,
                        "n_documents": n_documents, "seed": seed},
            justification=justification,
            summary={"n_candidates": len(candidates),
                     "n_chosen": len(subcorpus)},
            artifact=save_to)
    return subcorpus


def stage2_occurrences(subcorpus: Corpus, word: str,
                       strategy: str = "exhaustive",
                       n_total: Optional[int] = None,
                       key: Optional[Key] = None,
                       mode: str = MATCH_EXACT,
                       max_per_doc: Optional[int] = None,
                       min_distance: int = 0,
                       window_mode: str = WINDOW_WORDS,
                       window_size: int = 10,
                       seed: Optional[int] = None,
                       save_to: Optional[str] = None,
                       log: Optional[TraceLog] = None,
                       justification: str = "",
                       show_progress: bool = False) -> List[Unit]:
    """Extract the segment occurrences from the sub-corpus and add context.

    Steps: locate occurrences (with offset) -> per-document cap (optional) ->
    strategy -> KWIC window. Returns the final sample and saves it if asked.
    Records the decision in the log.

    If 'max_per_doc' is set and 'min_distance' > 0, the per-document cap keeps
    occurrences at least 'min_distance' characters apart (random but spaced),
    so they come from diverse contexts instead of clustering together.
    """
    occurrences = extract_segments(subcorpus, word, mode,
                                   show_progress=show_progress)
    if max_per_doc is not None:
        if min_distance > 0:
            occurrences = cap_per_group_spaced(
                occurrences, max_per_doc, lambda u: u["doc_id"],
                min_distance, seed, show_progress=show_progress)
        else:
            occurrences = cap_per_group(
                occurrences, max_per_doc, lambda u: u["doc_id"], seed,
                show_progress=show_progress)
    occurrences = _apply_strategy(occurrences, strategy, n_total, key, seed,
                                  show_progress=show_progress)
    concordance = [
        extract_window(subcorpus, occ, window_mode, window_size)
        for occ in _progress(occurrences, total=len(occurrences),
                             desc="Building KWIC windows", unit="occ",
                             enabled=show_progress)
    ]
    if save_to is not None:
        save_units(concordance, save_to, stage="stage2_occurrences")
    if log is not None:
        log.record(
            step=3, operation=f"stage2_occurrences[{strategy}]",
            parameters={"word": word, "mode": mode, "strategy": strategy,
                        "n_total": n_total, "max_per_doc": max_per_doc,
                        "min_distance": min_distance,
                        "window_mode": window_mode,
                        "window_size": window_size, "seed": seed},
            justification=justification,
            summary={"n_occurrences": len(concordance),
                     "n_articles": len({u["doc_id"] for u in concordance})},
            artifact=save_to)
    return concordance


def full_flow(corpus: Corpus, word: str,
              strategy_docs: str = "exhaustive",
              strategy_occ: str = "exhaustive",
              n_documents: Optional[int] = None,
              n_occurrences: Optional[int] = None,
              key_docs: Optional[Key] = None,
              key_occ: Optional[Key] = None,
              mode: str = MATCH_EXACT,
              max_per_doc: Optional[int] = None,
              window_mode: str = WINDOW_WORDS,
              window_size: int = 10,
              seed: Optional[int] = None,
              artifact_prefix: Optional[str] = None,
              log: Optional[TraceLog] = None,
              show_progress: bool = False) -> List[Unit]:
    """Run stage 1 and stage 2 in a row, with a different strategy per stage.

    If 'artifact_prefix' is given, saves sub-corpus and concordance with that
    prefix. Returns the final concordance.
    """
    art1 = f"{artifact_prefix}_subcorpus.json" if artifact_prefix else None
    art2 = f"{artifact_prefix}_kwic.json" if artifact_prefix else None
    subcorpus = stage1_documents(
        corpus, word, strategy_docs, n_documents, key_docs, mode, seed,
        save_to=art1, log=log,
        justification="Stage 1: select documents containing the segment.",
        show_progress=show_progress)
    return stage2_occurrences(
        subcorpus, word, strategy=strategy_occ, n_total=n_occurrences,
        key=key_occ, mode=mode, max_per_doc=max_per_doc,
        window_mode=window_mode, window_size=window_size, seed=seed,
        save_to=art2, log=log,
        justification="Stage 2: occurrences with context (KWIC).",
        show_progress=show_progress)


def print_kwic(concordance: List[Unit], width: int = 40) -> None:
    """Show the concordance in KWIC format aligned on the segment."""
    for u in concordance:
        left = u.get("left", "")[-width:].rjust(width)
        right = u.get("right", "")[:width].ljust(width)
        print(f"{u['doc_id'][:12]:12} | {left} [{u['text']}] {right}")
