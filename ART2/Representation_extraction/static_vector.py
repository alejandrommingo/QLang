"""
Static vector from the model's input embeddings.

A third kind of representation, alongside LSA (classic static) and the
contextual vectors (BERT/GPT run on text). Here we read a term's vector
directly from the model's INPUT EMBEDDING matrix -- the table the model uses
to turn token ids into vectors before any context is applied. So this is a
context-free vector for the term, taken from the model itself.

If the term is split into several sub-word tokens, their embedding vectors are
averaged. The vector can be L2-normalized.

Useful as a simple, cheap baseline (no need to run the full model on text).
Needs: transformers + torch.
"""

from __future__ import annotations

from typing import Optional


def get_static_vector(term: str, model_name: str = "bert-base-uncased",
                      normalize: bool = True, tokenizer=None, model=None):
    """Return the term's static vector from the model's input embeddings.

    Tokenizes the term (no special tokens), looks up each sub-token's row in
    the embedding matrix, and averages them. If 'normalize', the result is
    L2-normalized. 'tokenizer'/'model' can be injected for testing.

    Returns a numpy array (the vector).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "Needs 'transformers' and 'torch'. "
            "Install with: pip install transformers torch") from e

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if model is None:
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(term, add_special_tokens=False, return_tensors="pt")
    token_ids = enc["input_ids"][0].to(device)
    if token_ids.numel() == 0:
        raise ValueError(f"No tokens for term: {term!r}")

    emb_matrix = model.get_input_embeddings().weight  # vocab x hidden
    with torch.no_grad():
        subtoken_embs = emb_matrix[token_ids]         # n_subtokens x hidden
        vec = subtoken_embs.mean(dim=0)
        if normalize:
            vec = vec / (vec.norm(p=2) + 1e-12)
    return vec.detach().cpu().numpy()
