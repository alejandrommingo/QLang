"""
exe -- representation extraction.

Finds the tokenized sample produced by the tokenization step (in a sibling
folder), then runs the contextual model and turns it into vectors:

  fragment  ->  fit each unit to the model's token limit
  run model ->  per-token vectors from the chosen layer(s)
  aggregate ->  one vector per unit
  contour   ->  occurrences x dimensions matrix (for a target word)

It asks you, with menus, how to fragment, which layer(s) to use, and how to
aggregate, and whether to save the heavy per-token vectors. The aggregated
vectors and the contour are always saved.

Folder layout assumed (this folder is a sibling of the tokenization folder):

    parent/
      text_prep_model_tokenization/   (produces output/<run>/model_tokens.json)
      representation_extraction/       (this exe.py, fragmentation.py, ...)

Usage (from a terminal):
    python exe.py
"""

import os

import fragmentation as fr
import model_runner as mr
import aggregation as ag
import contour as ct
from tracelog import TraceLog

# names of the tokenized-sample file the tokenization step produces
TOKEN_FILENAMES = ["model_tokens.json"]


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


def progress_iter(iterable, total=None, desc="", unit="item"):
    """Wrap an iterable with tqdm for the interactive exe."""
    try:
        from tqdm import tqdm
    except ImportError:
        return simple_progress(iterable, total=total, desc=desc, unit=unit)
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def simple_progress(iterable, total=None, desc="", unit="item"):
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


# ---------------------------------------------------------------------------
# Finding the tokenized samples in the sibling folder
# ---------------------------------------------------------------------------

def find_tokenized():
    """Find tokenized-sample files, searching from this script's location.

    Looks in: this folder's output/, then sibling folders' output/, returning
    a list of (run_name, path). Works regardless of the current directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    found = []

    def scan(base):
        if not os.path.isdir(base):
            return
        for run in sorted(os.listdir(base)):
            run_dir = os.path.join(base, run)
            if not os.path.isdir(run_dir):
                continue
            for name in TOKEN_FILENAMES:
                p = os.path.join(run_dir, name)
                if os.path.exists(p):
                    found.append((run, p))
                    break

    scan(os.path.join(here, "output"))
    if not found:
        parent = os.path.dirname(here)
        for sibling in sorted(os.listdir(parent)):
            scan(os.path.join(parent, sibling, "output"))
    return found


def load_units(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content["data"] if isinstance(content, dict) else content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _static_path(units, model_name, out_dir, log):
    """Short path: one static vector per distinct target word, taken from the
    model's input embeddings (no context, no model run on text)."""
    import json
    import static_vector as sv

    # the target word(s) present in the sample
    words = sorted({u.get("text") for u in units if u.get("text")})
    print(f"\nComputing static vectors for: {words}")
    normalize = ask("L2-normalize the vectors? (Y/n)", "Y").lower() \
        .startswith("y")

    results = {}
    try:
        for w in progress_iter(words, total=len(words),
                               desc="Computing static vectors", unit="word"):
            vec = sv.get_static_vector(w, model_name=model_name,
                                       normalize=normalize)
            results[w] = vec.tolist()
    except Exception as e:
        print(f"\n  ERROR computing static vectors: {e}")
        print("  (check the model id and that 'transformers'/'torch' are "
              "installed)")
        return

    out_path = os.path.join(out_dir, "static_vectors.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"stage": "static_vectors", "model": model_name,
                               "normalized": normalize, "n_words": len(results)},
                   "data": results}, f, ensure_ascii=False, indent=2)
    log.record(
        step=5, operation="static_vectors",
        parameters={"model_name": model_name, "normalize": normalize},
        justification="Context-free term vectors from the model's input "
        "embeddings.",
        summary={"n_words": len(results), "model": model_name},
        artifact=out_path)

    log_path = os.path.join(out_dir, "tracelog_part3.json")
    log.save(log_path)
    print(f"\nStatic vectors saved: {out_path}")
    print(f"Tracelog: {log_path}")


def _model_max_tokens(model_name, fallback=512):
    """Read the model's maximum input length from its config.

    Uses the tokenizer's model_max_length when it is a sane value, otherwise
    the model config's max_position_embeddings, otherwise a safe fallback.
    This avoids asking the user for something the model already defines.
    """
    try:
        from transformers import AutoConfig, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        mml = getattr(tok, "model_max_length", None)
        if isinstance(mml, int) and 0 < mml < 100000:
            return mml
        cfg = AutoConfig.from_pretrained(model_name)
        mpe = getattr(cfg, "max_position_embeddings", None)
        if isinstance(mpe, int) and mpe > 0:
            return mpe
    except Exception:
        pass
    return fallback


def main():
    print("=" * 60)
    print("  REPRESENTATION EXTRACTION")
    print("=" * 60)

    samples = find_tokenized()
    if not samples:
        print("\nNo tokenized samples (model_tokens.json) found in this folder "
              "or a sibling 'output' folder. Run the tokenization step first.")
        return

    labels = [f"{run}   ({os.path.basename(p)})" for run, p in samples]
    chosen = choose("\nWhich tokenized sample do you want to process?",
                    labels, labels[0])
    run, token_path = samples[labels.index(chosen)]
    units = load_units(token_path)
    print(f"\nLoaded {len(units)} units from {token_path}")

    # model id: it MUST be the one used to tokenize, because the token ids
    # being loaded were produced by it. We read it from the tokens and use it
    # directly -- changing it would make the ids meaningless, so we don't ask.
    model_name = units[0].get("model_name") if units else None
    if not model_name:
        print("\n  The sample does not record which model tokenized it; "
              "cannot continue.")
        return
    print(f"\nUsing the model the sample was tokenized with: {model_name}")

    out_dir = os.path.join("output", run)
    os.makedirs(out_dir, exist_ok=True)
    log = TraceLog()

    # Choose the representation type.
    rep = choose(
        "\nWhich representation?",
        ["contextual (run the model on each occurrence's context)",
         "static (term vector from the model's embeddings, no context)"],
        "contextual (run the model on each occurrence's context)")

    if rep.startswith("static"):
        _static_path(units, model_name, out_dir, log)
        return

    # 1) FRAGMENTATION
    # the token limit is a property of the model, so read it automatically
    # instead of asking the user.
    max_tokens = _model_max_tokens(model_name)
    print(f"\nModel token limit (read from the model): {max_tokens}")
    strat = choose(
        "\nFragmentation strategy (for sentence/document units; segment units "
        "are always centered on the word)",
        [fr.SLIDING, fr.FIXED, fr.TRUNCATE], fr.SLIDING)
    window = stride = None
    if strat == fr.SLIDING:
        window = ask_int("Window size (tokens)", max_tokens)
        stride = ask_int("Stride (advance between windows)", max_tokens // 2)
    frag_path = (os.path.join(out_dir, "fragmented.json")
                 if ask("Save the fragmented units? (y/N)", "N").lower()
                 .startswith("y") else None)
    units = fr.fragment_units(units, max_tokens=max_tokens, strategy=strat,
                              window=window, stride=stride, save_to=frag_path,
                              show_progress=True)
    print(f"After fragmentation: {len(units)} units/fragments")

    # 2) RUN THE MODEL
    layer_mode = choose(
        "\nWhich layer(s) to extract?",
        [mr.LAYER_LAST, mr.LAYER_AVG_LAST_N], mr.LAYER_LAST)
    n_last = (ask_int("How many last layers to average", 4)
              if layer_mode == mr.LAYER_AVG_LAST_N else 4)
    batch_size = ask_int("Batch size", 16)
    save_tokens = ask(
        "Save the per-token vectors? They can be VERY large (y/N)", "N"
        ).lower().startswith("y")
    tok_path = os.path.join(out_dir, "token_vectors.json") if save_tokens \
        else None

    print(f"\nLoading and running '{model_name}'... "
          "(first time downloads the model weights)")
    try:
        units = mr.run_model(units, model_name, layer_mode=layer_mode,
                             n_last=n_last, batch_size=batch_size,
                             save_to=tok_path, log=log,
                             justification=f"Run {model_name}, layers="
                             f"{layer_mode}.",
                             show_progress=True)
    except Exception as e:
        print(f"\n  ERROR running the model: {e}")
        print("  (check the model id and that 'transformers' and 'torch' are "
              "installed)")
        return

    # 3) AGGREGATION
    agg = choose("\nAggregation strategy?",
                 [ag.MEAN, ag.REPRESENTATIVE], ag.MEAN)
    representative = "cls"
    if agg == ag.REPRESENTATIVE:
        representative = choose(
            "Representative token?",
            ["cls", "last"], "cls")
    agg_path = os.path.join(out_dir, "aggregated.json")
    units = ag.aggregate_units(units, strategy=agg,
                               representative=representative,
                               save_to=agg_path, log=log,
                               justification=f"Aggregate with {agg}.",
                               show_progress=True)
    n_vec = sum(1 for u in units if u.get("vector") is not None)
    print(f"\nAggregated: {n_vec} units have a vector. Saved: {agg_path}")

    # 4) CONTOUR (only meaningful for segment units)
    has_segments = any(u.get("unit_kind", "segment") == "segment"
                       for u in units)
    if has_segments:
        word = units[0].get("text")
        contour_path = os.path.join(out_dir, "contour.json")
        try:
            c = ct.build_contour(units, word=word, save_to=contour_path,
                                 log=log,
                                 justification="Contour of the target word "
                                 "across its contexts.",
                                 show_progress=True)
            print(f"\nContour built: {c['shape'][0]} occurrences x "
                  f"{c['shape'][1]} dimensions. Saved: {contour_path}")
        except ValueError as e:
            print(f"\n  Could not build contour: {e}")
    else:
        print("\nUnits are sentences/documents, so no word contour is built; "
              "the aggregated vectors are the output.")

    # tracelog
    log_path = os.path.join(out_dir, "tracelog_part3.json")
    log.save(log_path)
    print("\n" + "=" * 60)
    print("  TRACELOG")
    print("=" * 60)
    for i, e in enumerate(log.entries(), 1):
        print(f"  {i}. {e['operation']}  ->  {e['summary']}")
    print(f"\nOutputs in ./{out_dir}/")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {out_dir}/{name}")


if __name__ == "__main__":
    main()
