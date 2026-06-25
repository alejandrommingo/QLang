"""
Part 2 exe -- model preparation and representation.

Pipeline for the second folder of the project. It:
  1. finds the samples produced by Part 1 (in the sibling folder
     ../<part1>/samples/), and lets you pick one from a menu;
  2. applies minimal TEXT PREPARATION and saves it (prepared.json) -- the
     text BEFORE the model;
  3. lets you choose a representation:
       - contextual model (BERT / GPT-2 / any HF model): tokenize + align;
       - LSA (static): term-document matrix + SVD;
     and saves that output (model_tokens.json or lsa_space.json) -- the text
     AFTER loading the model / estimating LSA;
  4. saves the Part 2 tracelog.

Folder layout assumed:
    main_project/
      <part1>/   exe.py, corpus_loading.py, sampling.py, tracelog.py, samples/
      <part2>/   this exe.py, text_prep.py, tokenization.py, lsa.py, tracelog.py

Usage (run from inside the Part 2 folder):
    python exe.py
"""

import os

import text_prep as tp
import tokenization as tk
import lsa
from tracelog import TraceLog

# sample file names Part 1 can produce
SAMPLE_FILENAMES = ["stage2_kwic.json", "occurrences_kwic.json",
                    "segment_sample.json", "sentence_sample.json",
                    "document_sample.json"]


def ask(text, default=None):
    suffix = f" [{default}]" if default is not None else ""
    r = input(f"{text}{suffix}: ").strip()
    return r if r else (default if default is not None else "")


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


# ---------------------------------------------------------------------------
# Finding the Part 1 samples in the sibling folder
# ---------------------------------------------------------------------------

def find_samples_dir():
    """Locate the 'samples' folder produced by Part 1.

    Searches relative to THIS script's location (not the working directory),
    so it works no matter which folder you launch it from. Looks, in order:
      1. <script_dir>/samples
      2. <script_dir>/../*/samples      (sibling folders of Part 2)
      3. <parent_of_script>/samples      (samples sitting in the main project)
    Returns the path to the samples dir, or None.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # 1. inside the script's own folder
    local = os.path.join(here, "samples")
    if os.path.isdir(local):
        return local

    # 2. sibling folders of the script's folder (the usual case)
    parent = os.path.dirname(here)
    for sibling in sorted(os.listdir(parent)):
        candidate = os.path.join(parent, sibling, "samples")
        if os.path.isdir(candidate):
            return candidate

    # 3. samples directly in the parent (main project) folder
    direct = os.path.join(parent, "samples")
    if os.path.isdir(direct):
        return direct

    return None


def find_samples(samples_dir):
    """Return [(run_name, sample_file_path)] for each run with a sample."""
    found = []
    for run in sorted(os.listdir(samples_dir)):
        run_dir = os.path.join(samples_dir, run)
        if not os.path.isdir(run_dir):
            continue
        for name in SAMPLE_FILENAMES:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                found.append((run, path))
                break
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PART 2 - MODEL PREPARATION AND REPRESENTATION")
    print("=" * 60)

    samples_dir = find_samples_dir()
    if samples_dir is None:
        print("\nCould not find a 'samples' folder automatically. Expected it "
              "in this folder or in a sibling folder of the main project.")
        samples_dir = ask("Type the path to the samples folder (or leave "
                          "empty to cancel)", "")
        if not samples_dir or not os.path.isdir(samples_dir):
            print("No valid samples folder. Run Part 1 (exe.py) first.")
            return
    else:
        # confirm the auto-detected folder and let the user override it
        print(f"\nSamples folder found: {samples_dir}")
        other = ask("Press Enter to use it, or type another path", "")
        if other:
            if os.path.isdir(other):
                samples_dir = other
            else:
                print(f"  '{other}' is not a folder; keeping {samples_dir}")
    print(f"\nUsing samples from: {samples_dir}")

    samples = find_samples(samples_dir)
    if not samples:
        print("No samples found inside that folder.")
        return

    # 1) pick a sample
    labels = [f"{run}   ({os.path.basename(p)})" for run, p in samples]
    chosen = choose("\nWhich sample do you want to process?",
                    labels, labels[0])
    run, sample_path = samples[labels.index(chosen)]

    # outputs go into a local 'output/<run>/' folder in Part 2
    out_dir = os.path.join("Text_prep_model_tokenization","output", run)
    os.makedirs(out_dir, exist_ok=True)
    log = TraceLog()

    # 2) TEXT PREPARATION (before the model)
    level = choose(
        "\nText preparation level (the article recommends minimal):",
        ["safe only (whitespace + Unicode) - recommended",
         "also lowercase (RISKY)",
         "also strip accents (RISKY)",
         "lowercase + strip accents (RISKY)"],
        "safe only (whitespace + Unicode) - recommended")
    steps = list(tp.SAFE_DEFAULTS)
    if "lowercase" in level:
        steps.append("lowercase")
    if "strip" in level:
        steps.append("strip_accents")

    prepared_path = os.path.join(out_dir, "prepared.json")
    prepared = tp.prepare_units(
        sample_path, steps=steps, save_to=prepared_path, log=log,
        justification="Minimal text preparation before the model.")
    print(f"\nText prepared and saved: {prepared_path}  ({len(prepared)} units)")

    # 3) REPRESENTATION: contextual model or LSA
    approach = choose(
        "\nWhich representation do you want?",
        ["contextual model (BERT / GPT-2 / other) - tokenize + align",
         "LSA (static) - term-document matrix + SVD"],
        "contextual model (BERT / GPT-2 / other) - tokenize + align")

    if approach.startswith("contextual"):
        # menu of common models, so you don't have to type the HF id.
        # grouped by use; ids verified on the Hugging Face hub.
        model_menu = [
            "BERT base, uncased (bert-base-uncased) - English, general",
            "BERT base, cased (bert-base-cased) - English, keeps casing",
            "RoBERTa base (roberta-base) - English, stronger than BERT",
            "DistilBERT (distilbert-base-uncased) - English, smaller/faster",
            "GPT-2 (gpt2) - decoder, for surprisal",
            "BETO (dccuchile/bert-base-spanish-wwm-cased) - Spanish",
            "BERT multilingual (bert-base-multilingual-cased) - many langs",
            "XLM-RoBERTa (xlm-roberta-base) - strong multilingual",
            "MPNet sentence (sentence-transformers/all-mpnet-base-v2) - "
            "best for meaning similarity, per-sentence",
            "other (type the Hugging Face id)",
        ]
        model_ids = {
            model_menu[0]: "bert-base-uncased",
            model_menu[1]: "bert-base-cased",
            model_menu[2]: "roberta-base",
            model_menu[3]: "distilbert-base-uncased",
            model_menu[4]: "gpt2",
            model_menu[5]: "dccuchile/bert-base-spanish-wwm-cased",
            model_menu[6]: "bert-base-multilingual-cased",
            model_menu[7]: "xlm-roberta-base",
            model_menu[8]: "sentence-transformers/all-mpnet-base-v2",
        }
        picked = choose("\nWhich model?", model_menu, model_menu[0])
        if picked.startswith("other"):
            model_name = ask("Hugging Face model id", tk.DEFAULT_MODEL)
        else:
            model_name = model_ids[picked]

        print(f"\nLoading tokenizer '{model_name}' and aligning... "
              "(first time downloads the tokenizer)")
        tok_path = os.path.join(out_dir, "model_tokens.json")
        try:
            tk.tokenize_units(
                prepared, model_name=model_name, save_to=tok_path, log=log,
                justification=f"Tokenize and align with {model_name}.")
            print(f"Tokenized + aligned, saved: {tok_path}")
        except Exception as e:
            print(f"\n  ERROR loading/tokenizing: {e}")
            print("  (check the model id and that 'transformers' is installed)")
    else:
        n_comp = int(ask("Number of LSA dimensions", "100"))
        docs = lsa.documents_from_prepared(prepared)
        lsa_path = os.path.join(out_dir, "lsa_space.json")
        try:
            res = lsa.estimate_lsa(
                docs, n_components=n_comp, save_to=lsa_path, log=log,
                justification="Static LSA representation over the documents.")
            print(f"\nLSA estimated: {res['n_components']} dims, "
                  f"{res['vocabulary_size']} words, saved: {lsa_path}")
        except Exception as e:
            print(f"\n  ERROR estimating LSA: {e}")

    # 4) save Part 2 tracelog
    log_path = os.path.join(out_dir, "tracelog_part2.json")
    log.save(log_path)
    print("\n" + "=" * 60)
    print("  TRACELOG (Part 2)")
    print("=" * 60)
    for i, e in enumerate(log.entries(), 1):
        print(f"  {i}. [step {e['step']}] {e['operation']}")
        print(f"     why    : {e['justification']}")
        print(f"     summary: {e['summary']}")
    print(f"\nOutputs saved in ./{out_dir}/")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {out_dir}/{name}")


if __name__ == "__main__":
    main()
