"""
Asks questions on the console and runs the sampling for the unit you choose
(segment, sentence or document), saving every artifact to its own folder so
that successive runs do not overwrite each other:

    samples/<term>_<unit>/

For example, running with "china" as a segment writes to
samples/china_segment/, and a later run with "north korea" writes to
samples/north_korea_segment/.

Usage:
    python demo.py

At each prompt, pressing Enter uses the default shown in [brackets].
"""

import os
import re

import corpus_loading as cl
import sampling as sp
from tracelog import TraceLog


# ===========================================================================
# Console helpers
# ===========================================================================

def ask(text, default=None):
    suffix = f" [{default}]" if default is not None else ""
    r = input(f"{text}{suffix}: ").strip()
    return r if r else (default if default is not None else "")


def ask_int(text, default):
    while True:
        r = ask(text, default)
        try:
            return int(r)
        except (ValueError, TypeError):
            print("  (please type an integer)")


def choose(text, options, default):
    print(text)
    for i, op in enumerate(options, 1):
        mark = " (default)" if op == default else ""
        print(f"  {i}. {op}{mark}")
    while True:
        r = ask("Choose a number", options.index(default) + 1)
        try:
            idx = int(r) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, TypeError):
            pass
        print("  (number out of range)")


def slugify(text):
    """Turn a term into a safe folder name: 'North Korea' -> 'north_korea'."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s)
    return s or "term"


# ===========================================================================
# Step 1 -- get the corpus
# ===========================================================================

def get_corpus(search_term, default_lang="en"):
    """Ask where the corpus comes from: Wikipedia (API or dataset) or manual."""
    origin = choose("\nWhere does the corpus come from?",
                    ["wikipedia", "manual"], "wikipedia")

    if origin == "wikipedia":
        via = choose(
            "How should the Wikipedia articles be fetched?",
            ["dataset (large-scale sampling, recommended)",
             "api (a few articles, faster for testing)"],
            "dataset (large-scale sampling, recommended)")
        lang = ask("Language code (en or es)", default_lang)
        if not search_term:
            search_term = ask("Term to search articles in Wikipedia", "")

        if via.startswith("dataset"):
            return _corpus_dataset(search_term, lang)
        return _corpus_api(search_term, lang)

    return _corpus_manual(), "manual", "en"


def _corpus_dataset(term, lang):
    """Dataset path (Hugging Face): large-scale sampling (no 429)."""
    print("\nFirst, how many Wikipedia articles to bring INTO THE CORPUS.")
    print("(This is the raw material; the actual sampling comes later.)")
    k = ask_int("How many articles to fetch into the corpus", 300)
    min_occ = ask_int("Minimum times the word must appear in an article", 2)
    max_iter = ask_int(
        "How many Wikipedia articles to scan while fetching "
        "(higher = more thorough but slower)", 200000)
    seed = ask_int("Random seed", 123)
    print(f"\nSampling '{term}' over Wikipedia ({lang}) via dataset...")
    print("  (the first time it downloads the dataset; this may take a while)")
    try:
        corpus = cl.sample_wikipedia_dataset(
            term, lang=lang, k=k, seed=seed, max_iter=max_iter,
            min_occurrences=min_occ, progress_every=20000)
    except Exception as e:
        print(f"\n  ERROR with the dataset: {e}")
        op = choose("What now?",
                    ["retry", "use the web API", "manual corpus"], "retry")
        if op == "retry":
            return _corpus_dataset(term, lang)
        if op == "use the web API":
            return _corpus_api(term, lang)
        return _corpus_manual(), "manual", "en"
    print(f"  sampled {len(corpus)} articles.")
    return corpus, "wikipedia", lang


def _corpus_api(term, lang):
    """Web API path: fetch articles mentioning the term (a few)."""
    n = ask_int("How many articles to fetch at most?", 15)
    print(f"\nSearching Wikipedia ({lang}) for articles about '{term}'...")
    try:
        corpus = cl.search_and_load_wikipedia(term, lang=lang, n=n)
    except Exception as e:
        print(f"\n  ERROR querying the API: {e}")
        op = choose("What now?",
                    ["retry", "use the dataset", "manual corpus"], "retry")
        if op == "retry":
            return _corpus_api(term, lang)
        if op == "use the dataset":
            return _corpus_dataset(term, lang)
        return _corpus_manual(), "manual", "en"
    if not corpus:
        print("  No articles found for that term.")
    print(f"  fetched {len(corpus)} articles.")
    return corpus, "wikipedia", lang


def _corpus_manual():
    """Ask documents on the console and return a corpus."""
    print("\nEnter documents. Empty line to finish.")
    corpus = {}
    i = 1
    while True:
        text = input(f"  text {i} (empty to stop): ").strip()
        if not text:
            break
        source = input("    source (e.g. wikipedia): ").strip() or "manual"
        corpus[f"doc_{i}"] = cl.make_document(
            text, {"source": source, "language": "en"})
        i += 1
    return corpus


def ensure_minimum_wikipedia(corpus, word, mode, min_docs, lang):
    """Fetch more Wikipedia articles via API if the minimum is not reached."""
    def n_with(c):
        return len(sp.select_documents_with_word(c, word, mode))

    if n_with(corpus) >= min_docs:
        return corpus

    print(f"\nMinimum of {min_docs} documents with '{word}' not reached. "
          f"Extending the Wikipedia search...")
    try:
        more = cl.search_and_load_wikipedia(
            word, lang=lang, n=max(min_docs * 3, 20), exclude=set(corpus))
    except Exception as e:
        print(f"  (could not extend: {e})")
        return corpus

    for doc_id, doc in more.items():
        corpus.setdefault(doc_id, doc)
    got = n_with(corpus)
    if got < min_docs:
        print(f"  warning: only {got} documents with the word; continuing.")
    else:
        print(f"  now there are {got} documents with the word.")
    return corpus


# ===========================================================================
# Auditing
# ===========================================================================

def save_corpus_and_show(corpus, log, out_dir):
    path = os.path.join(out_dir, "corpus_full.json")
    cl.save_corpus(corpus, path)
    print(f"\nCorpus ({len(corpus)} docs) saved to {path}")
    for doc_id, doc in corpus.items():
        print(f"  {doc_id:18} | {doc['meta']} | {doc['body'][:40]}...")
    log.record(step=1, operation="load_corpus",
               summary={"n_documents": len(corpus)},
               justification="Starting corpus.", artifact=path)


def add_reference(units):
    """Attach a readable reference to the source article to each unit."""
    for u in units:
        u["reference"] = {
            "doc_id": u["doc_id"],
            "title": u["meta"].get("title", u["doc_id"]),
            "source": u["meta"].get("source"),
        }
    return units


# ===========================================================================
# Segment case (one or two stages, chosen by the user)
# ===========================================================================

def _needs_count(strategy):
    """Strategies that require an explicit number of items."""
    return strategy in ("random", "stratified", "reservoir")


def choose_strategy(prompt, corpus):
    """Ask which sampling strategy to use.

    Offers 'stratified' only when there is more than one source to stratify
    by; otherwise it would be meaningless. Returns (strategy, key) where key
    is the grouping function for stratified (or None).
    """
    sources = {d["meta"].get("source") for d in corpus.values()}
    options = ["exhaustive", "random", "reservoir"]
    if len(sources) > 1:
        options.insert(2, "stratified")
    strat = choose(prompt, options, "exhaustive")
    key = (lambda u: u["meta"].get("source")) if strat == "stratified" else None
    return strat, key


def case_segment(corpus, word, mode, cfg, lang, origin, log, out_dir):
    if cfg["only_source"]:
        corpus = filter_source(corpus, cfg["only_source"])
        print(f"\nFiltered to source '{cfg['only_source']}': {list(corpus)}")

    is_api = any(d["meta"].get("via") == "api" for d in corpus.values())
    if origin == "wikipedia" and is_api:
        corpus = ensure_minimum_wikipedia(
            corpus, word, mode, cfg["min_docs"], lang)

    if cfg["two_stages"]:
        _segment_two_stages(corpus, word, mode, cfg, log, out_dir)
    else:
        _segment_one_stage(corpus, word, mode, cfg, log, out_dir)


def _segment_two_stages(corpus, word, mode, cfg, log, out_dir):
    """Stage 1 (choose documents) then stage 2 (occurrences + window)."""
    sub_path = os.path.join(out_dir, "stage1_subcorpus.json")
    subcorpus = sp.stage1_documents(
        corpus, word,
        strategy=cfg["strategy_docs"], n_documents=cfg["max_docs"],
        key=cfg["key_docs"], mode=mode, seed=cfg["seed"],
        save_to=sub_path, log=log,
        justification=f"Documents with '{word}' "
                      f"({cfg['strategy_docs']}, max {cfg['max_docs']}).")
    print(f"\nSTAGE 1: {len(subcorpus)} documents -> {list(subcorpus)}")

    _warn_if_cap_blocks_target(len(subcorpus), cfg)

    kwic_path = os.path.join(out_dir, "stage2_kwic.json")
    conc = sp.stage2_occurrences(
        subcorpus, word,
        strategy=cfg["strategy_occ"], n_total=cfg["max_occ"],
        key=cfg["key_occ"], mode=mode, max_per_doc=cfg["max_per_doc"],
        min_distance=cfg.get("min_distance", 0),
        window_mode=cfg["window"], window_size=cfg["window_size"],
        seed=cfg["seed"], save_to=kwic_path, log=log,
        justification=(f"Occurrences ({cfg['strategy_occ']}) with "
                       f"'{cfg['window']}' window; "
                       f"max {cfg['max_per_doc']} per document."))
    _report_concordance(conc, kwic_path, cfg)


def _segment_one_stage(corpus, word, mode, cfg, log, out_dir):
    """Single stage: all occurrences in the corpus -> strategy -> window."""
    occurrences = sp.extract_segments(corpus, word, mode)
    n_articles_with_word = len({u["doc_id"] for u in occurrences})
    _warn_if_cap_blocks_target(n_articles_with_word, cfg)
    if cfg["max_per_doc"] is not None:
        if cfg.get("min_distance", 0) > 0:
            occurrences = sp.cap_per_group_spaced(
                occurrences, cfg["max_per_doc"], lambda u: u["doc_id"],
                cfg["min_distance"], cfg["seed"])
        else:
            occurrences = sp.cap_per_group(
                occurrences, cfg["max_per_doc"], lambda u: u["doc_id"],
                cfg["seed"])
    occurrences = sp._apply_strategy(
        occurrences, cfg["strategy_occ"], cfg["max_occ"],
        cfg["key_occ"], cfg["seed"])
    conc = [sp.extract_window(corpus, occ, cfg["window"], cfg["window_size"])
            for occ in occurrences]
    kwic_path = os.path.join(out_dir, "occurrences_kwic.json")
    sp.save_units(conc, kwic_path, stage="single_stage_occurrences")
    log.record(
        step=3, operation=f"single_stage[{cfg['strategy_occ']}]",
        parameters={"word": word, "mode": mode,
                    "strategy": cfg["strategy_occ"], "max_occ": cfg["max_occ"],
                    "max_per_doc": cfg["max_per_doc"], "seed": cfg["seed"]},
        justification=f"All occurrences of '{word}' in one stage "
                      f"({cfg['strategy_occ']}).",
        summary={"n_occurrences": len(conc),
                 "n_articles": len({u['doc_id'] for u in conc})},
        artifact=kwic_path)
    _report_concordance(conc, kwic_path, cfg)


def _warn_if_cap_blocks_target(n_articles, cfg):
    """Warn (clearly, before processing) if the per-article cap makes the
    requested number of occurrences impossible to reach.

    The ceiling of total occurrences is n_articles * max_per_doc. If the
    requested minimum/maximum is above that ceiling, no run can reach it, so we
    say so and suggest how many articles (or what cap) would be needed.
    """
    cap = cfg.get("max_per_doc")
    if not cap:
        return  # no per-article cap -> no ceiling from this
    target = cfg.get("max_occ") or cfg.get("min_occ")
    if not target:
        return
    ceiling = n_articles * cap
    if ceiling < target:
        needed_articles = -(-target // cap)  # ceil division
        needed_cap = -(-target // n_articles)
        print("\n" + "!" * 60)
        print(f"  HEADS UP: you asked for {target} occurrences, but with "
              f"{n_articles} articles")
        print(f"  and a cap of {cap} per article the most you can get is "
              f"{n_articles} x {cap} = {ceiling}.")
        print(f"  To reach {target} you would need either:")
        print(f"    - at least {needed_articles} articles "
              f"(keeping {cap} per article), or")
        print(f"    - a cap of at least {needed_cap} per article "
              f"(with {n_articles} articles).")
        print(f"  The run will continue and give you up to {ceiling}.")
        print("!" * 60)


def _report_concordance(conc, kwic_path, cfg):
    """Add reference, re-save, and print the concordance + spread."""
    conc = add_reference(conc)
    sp.save_units(conc, kwic_path, stage="occurrences")
    if len(conc) < cfg["min_occ"]:
        print(f"\nNote: a minimum of {cfg['min_occ']} occurrences was asked "
              f"and there are only {len(conc)}. Continuing with what we have.")
    print(f"\n{len(conc)} occurrences")
    print("\nConcordance (segment with its context):")
    for u in conc:
        line = (u["left"] + "[" + u["text"] + "]" + u["right"]
                ).replace("\n", " ")
        print(f"  [{u['reference']['title']}] {line[:100]}")
    print(f"\nSpread per document: "
          f"{sp.summary_per_group(conc, lambda u: u['doc_id'])}")


# ===========================================================================
# Sentence or document case
# ===========================================================================

def case_sentence_or_doc(corpus, unit, expression, cfg, log, out_dir):
    if cfg["only_source"]:
        corpus = filter_source(corpus, cfg["only_source"])
        print(f"\nFiltered to source '{cfg['only_source']}': {list(corpus)}")

    units = (sp.extract_sentences(corpus) if unit == "sentence"
             else sp.extract_documents(corpus))

    if expression:
        pattern = sp.build_pattern(expression, sp.MATCH_LOOSE)
        units = [u for u in units if pattern.search(u["text"])]

    if cfg["max_occ"]:
        units = sp._apply_strategy(
            units, cfg["strategy_occ"], cfg["max_occ"],
            cfg["key_occ"], cfg["seed"])

    units = add_reference(units)
    path = os.path.join(out_dir, f"{unit}_sample.json")
    sp.save_units(units, path, stage=f"sample[{unit}]")
    log.record(
        step=3, operation=f"sample[{unit}]",
        parameters={"expression": expression, "max_occ": cfg["max_occ"]},
        justification=f"Unit = {unit}" +
                      (f" containing '{expression}'." if expression else "."),
        summary={"n_units": len(units),
                 "n_articles": len({u["doc_id"] for u in units})},
        artifact=path)

    if len(units) < cfg["min_occ"]:
        print(f"\nNote: minimum {cfg['min_occ']} and only {len(units)}. "
              f"Continuing.")

    print(f"\n{len(units)} units selected:")
    for u in units:
        print(f"  [{u['reference']['title']}] offset={u['offset']} -> "
              f"{u['text'][:70]!r}")


def filter_source(corpus, source):
    """Keep only documents whose source matches (tolerant by prefix)."""
    cl.validate_corpus(corpus)
    return {
        doc_id: doc for doc_id, doc in corpus.items()
        if (doc["meta"].get("source") == source
            or str(doc["meta"].get("source", "")).startswith(source))
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("  INTERACTIVE SAMPLING DEMO")
    print("=" * 60)

    log = TraceLog()

    # 1) First: what to sample (unit) and the target.
    unit = choose("\nWhich unit do you want to sample?",
                  ["segment", "sentence", "document"], "segment")

    if unit == "segment":
        target = ask("Word/segment to search", "yellow")
        mode = choose("Match type",
                      [sp.MATCH_EXACT, sp.MATCH_VARIANTS, sp.MATCH_LOOSE],
                      sp.MATCH_EXACT)
    elif unit == "sentence":
        target = ask("Expression the sentence must contain (empty = all)",
                     "is a democratic country")
        mode = sp.MATCH_EXACT
    else:
        target = ask("Expression the document must contain (empty = all)", "")
        mode = sp.MATCH_EXACT

    # output folder per term and unit, so runs don't overwrite each other
    label = slugify(target) if target else "all"
    out_dir = os.path.join("Corpus_loading_and_sampling", "samples", f"{label}_{unit}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput folder for this run: {out_dir}/")

    # 2) Then: the corpus. If Wikipedia, it searches by the target.
    corpus, origin, lang = get_corpus(target)
    save_corpus_and_show(corpus, log, out_dir)

    # 3) Filters and limits. The source filter only appears if the corpus
    # mixes sources; with a single source it is not even asked.
    sources = {d["meta"].get("source") for d in corpus.values()}
    if len(sources) > 1:
        print(f"\nThe corpus has several sources: "
              f"{sorted(s for s in sources if s)}")
        only_source = ask("Restrict to one of them? (empty = all)", "")
    else:
        only_source = ""
    print("\nNow, of the articles that contain your word, how many to keep "
          "for the sample.")
    cfg = {
        "only_source": only_source or None,
        "min_docs": ask_int(
            "Minimum articles to keep (fewest acceptable)", 1),
        "max_docs": ask_int(
            "Maximum articles to keep (0 = keep all that have the word)",
            0) or None,
        "min_occ": ask_int("Minimum occurrences to keep (fewest acceptable)",
                           1),
        "max_occ": ask_int(
            "Maximum occurrences to keep (0 = no cap)", 0) or None,
        "max_per_doc": ask_int(
            "Maximum occurrences per article (0 = no cap)", 0) or None,
        "seed": ask_int("Random seed (reproducibility)", 0),
    }

    # Sampling stages and strategy.
    # Defaults so every case has the keys it needs.
    cfg["two_stages"] = False
    cfg["strategy_docs"] = "exhaustive"
    cfg["key_docs"] = None
    cfg["strategy_occ"] = "exhaustive"
    cfg["key_occ"] = None

    if unit == "segment":
        stages = choose(
            "\nSampling approach for the segment?",
            ["two stages (choose articles, then occurrences)",
             "one stage (all occurrences at once)"],
            "two stages (choose articles, then occurrences)")
        cfg["two_stages"] = stages.startswith("two")
        if cfg["two_stages"]:
            cfg["strategy_docs"], cfg["key_docs"] = choose_strategy(
                "Strategy for STAGE 1 (selecting articles)", corpus)
            if _needs_count(cfg["strategy_docs"]) and not cfg["max_docs"]:
                print(f"\n'{cfg['strategy_docs']}' draws a fixed number of "
                      f"articles at random, but you left 'maximum articles' "
                      f"as 0 (all). Tell it how many to draw:")
                cfg["max_docs"] = ask_int("Number of articles to draw", 100)
            cfg["strategy_occ"], cfg["key_occ"] = choose_strategy(
                "Strategy for STAGE 2 (selecting occurrences)", corpus)
        else:
            cfg["strategy_occ"], cfg["key_occ"] = choose_strategy(
                "Sampling strategy", corpus)
    else:
        cfg["strategy_occ"], cfg["key_occ"] = choose_strategy(
            "Sampling strategy", corpus)

    if _needs_count(cfg["strategy_occ"]) and not cfg["max_occ"]:
        print(f"\n'{cfg['strategy_occ']}' draws a fixed number of occurrences "
              f"at random, but you left 'maximum occurrences' as 0. "
              f"Tell it how many to draw:")
        cfg["max_occ"] = ask_int("Number of occurrences to draw", 100)
    # Ask the context window FIRST (it defines how much context you keep),
    # then the minimum distance between occurrences (which depends on it).
    if unit == "segment":
        cfg["window"] = choose(
            "Context window",
            [sp.WINDOW_CHARS, sp.WINDOW_WORDS, sp.WINDOW_SENTENCE,
             sp.WINDOW_PARAGRAPH], sp.WINDOW_SENTENCE)
        cfg["window_size"] = (
            ask_int("Window size (chars/words)", 50)
            if cfg["window"] in (sp.WINDOW_CHARS, sp.WINDOW_WORDS) else 0)

    # If there is a per-article cap, occurrences are kept spaced apart. The
    # minimum distance is computed AUTOMATICALLY from the window (so the
    # context windows of two occurrences do not overlap), and relaxed later
    # for short articles. We just report the value used.
    if cfg["max_per_doc"]:
        if unit == "segment" and cfg["window"] == sp.WINDOW_CHARS:
            cfg["min_distance"] = max(2 * cfg["window_size"], 100)
        elif unit == "segment" and cfg["window"] == sp.WINDOW_WORDS:
            cfg["min_distance"] = max(2 * cfg["window_size"] * 6, 100)
        else:
            cfg["min_distance"] = 500
        print(f"\nOccurrences of the same article will be kept at least "
              f"{cfg['min_distance']} characters apart (auto, relaxed for "
              f"short articles).")

    # 4) Run.
    if unit == "segment":
        case_segment(corpus, target, mode, cfg, lang, origin, log, out_dir)
    else:
        case_sentence_or_doc(corpus, unit, target, cfg, log, out_dir)

    # final audit
    log_path = os.path.join(out_dir, "tracelog.json")
    log.save(log_path)
    print("\n" + "=" * 60)
    print("  TRACELOG (auditable record)")
    print("=" * 60)
    for i, e in enumerate(log.entries(), 1):
        print(f"  {i}. [step {e['step']}] {e['operation']}")
        print(f"     why     : {e['justification']}")
        print(f"     summary : {e['summary']}")
        print(f"     file    : {e['artifact']}")
    print(f"\nEverything saved in ./{out_dir}/")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {out_dir}/{name}")


if __name__ == "__main__":
    main()
