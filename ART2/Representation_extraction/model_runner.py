"""
Running the model and extracting hidden states (per-token vectors).

This is where the contextual model is actually executed (its full weights, not
just the tokenizer). For each unit it produces, per token, a vector taken from
the chosen layer(s). The next module aggregates those token vectors into one
vector per unit.

What this module handles, following the methodology:

  - BATCHING: sequences are processed in groups (batch size, typically 8-64,
    limited by available memory).
  - PADDING + ATTENTION MASKS: within a batch, shorter sequences are padded to
    a common length with empty tokens, and an attention mask marks which
    positions are real text vs. padding, so padding does not affect the
    output.
  - LAYER SELECTION: lower layers capture surface features, middle ones
    syntax, upper ones semantics. For bidirectional models (BERT) a sensible
    start is the last layer or the average of the last few; for autoregressive
    ones (GPT) the last layer. The choice is recorded.

Input: tokenized (and possibly fragmented) units, each with 'token_ids'.
Output: each unit gains 'token_vectors' (one vector per token, from the chosen
layer or averaged layers). Needs: transformers + torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

Unit = Dict[str, Any]

# layer-selection modes
LAYER_LAST = "last"            # the final layer
LAYER_AVG_LAST_N = "avg_last_n"  # average of the last N layers


def load_model(model_name: str):
    """Load the full model (weights) for a Hugging Face id.

    Kept separate so the (heavy, possibly slow) download happens in one place
    and so it is easy to mock in tests. Requests hidden states from all layers.
    """
    try:
        from transformers import AutoModel
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Running the model needs 'transformers' and 'torch'. "
            "Install with: pip install transformers torch") from e
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.config.output_hidden_states = True
    model.eval()
    return model


def load_tokenizer_for_model(model_name: str, model=None):
    """Load a tokenizer ready for running the model, fixing padding.

    Some autoregressive models (e.g. GPT-2) have no padding token, which breaks
    batching. We set the padding token to the end-of-sequence token, fix the
    padding side, and tell the model's config which id is padding. This mirrors
    the standard GPT-2 workaround.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Needs 'transformers'. Install with: pip install transformers"
            ) from e
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


def _select_layer_vectors(hidden_states, layer_mode, n_last):
    """From the model's hidden_states (tuple: embeddings + one per layer),
    return a single per-token tensor according to the layer choice."""
    import torch
    # hidden_states[0] is the embedding layer; [1:] are the transformer layers
    if layer_mode == LAYER_LAST:
        return hidden_states[-1]
    if layer_mode == LAYER_AVG_LAST_N:
        last_n = hidden_states[-n_last:]
        return torch.stack(last_n, dim=0).mean(dim=0)
    raise ValueError(f"unknown layer mode: {layer_mode}")


def run_model(units: List[Unit], model_name: str,
              layer_mode: str = LAYER_LAST, n_last: int = 4,
              batch_size: int = 16, model=None, tokenizer=None,
              save_to: Optional[str] = None,
              log=None, justification: str = "") -> List[Unit]:
    """Run the model over the units and attach per-token vectors.

    Processes units in batches of 'batch_size'. Within each batch, pads
    sequences to equal length and builds attention masks so padding is
    ignored. Extracts vectors from the chosen layer(s). Each unit gains
    'token_vectors' (list of per-token vectors, aligned with its tokens) and
    records the layer choice.

    'model'/'tokenizer' can be injected (for testing); otherwise they are
    loaded from 'model_name'. 'save_to' is OPTIONAL and writes the heavy
    per-token vectors -- only pass it if you really want them on disk, since
    this file can be very large.
    """
    import torch

    if model is None:
        model = load_model(model_name)
    if tokenizer is None:
        tokenizer = load_tokenizer_for_model(model_name, model)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # some models (e.g. GPT-2) have no pad token; use EOS as padding
        pad_id = tokenizer.eos_token_id

    out: List[Unit] = []
    for start in range(0, len(units), batch_size):
        batch = units[start:start + batch_size]
        id_lists = [u["token_ids"] for u in batch]
        max_len = max(len(ids) for ids in id_lists)

        # pad each sequence to max_len and build attention masks
        input_ids = []
        attention = []
        for ids in id_lists:
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)

        input_ids = torch.tensor(input_ids)
        attention = torch.tensor(attention)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention)
        layer = _select_layer_vectors(outputs.hidden_states, layer_mode,
                                      n_last)  # (batch, max_len, hidden)

        for i, u in enumerate(batch):
            n = len(u["token_ids"])  # real tokens, ignore padding
            vecs = layer[i, :n, :].tolist()
            v = dict(u)
            v["token_vectors"] = vecs
            v["layer_mode"] = layer_mode
            if layer_mode == LAYER_AVG_LAST_N:
                v["n_last_layers"] = n_last
            out.append(v)

    if save_to is not None:
        _save_token_vectors(out, save_to, model_name, layer_mode)
    if log is not None:
        log.record(
            step=5, operation="run_model",
            parameters={"model_name": model_name, "layer_mode": layer_mode,
                        "n_last": n_last, "batch_size": batch_size},
            justification=justification or
            "Run the model and extract per-token vectors from the chosen "
            "layer(s).",
            summary={"n_units": len(out), "model": model_name,
                     "layer_mode": layer_mode},
            artifact=save_to)
    return out


def _save_token_vectors(units, path, model_name, layer_mode):
    """Save the heavy per-token vectors (optional; can be large)."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"stage": "token_vectors", "model": model_name,
                               "layer_mode": layer_mode, "n_units": len(units)},
                   "data": units}, f, ensure_ascii=False, indent=2)
