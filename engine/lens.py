"""Real-model logit lens + J-lens over HuggingFace causal LMs, measured in bits.

Adapted from x-logit-lens's ``lib/logit_lens.py`` (the classic lens is the same
~15 readable lines), then made MDL-native: every layer reports the code length
``-log2 p`` of the model's final prediction, so the panel shares the bits axis
with every other lens — "the prediction sharpens with depth" becomes "the code
shortens with depth".

The J-lens is the mini-GPT lens's treatment carried to a real model. The classic
lens decodes early residuals *as if the network were done*; the J-lens decodes
``J.h`` — the rung's content transported into the final logit basis by the
Jacobian of the remaining computation. On a real model the full Jacobian is out
of reach, but ``J.h`` for a single direction is one forward-mode AD pass
(a JVP): seed block ``l``'s input with tangent = its own last-position content
and read the tangent off the logits. One extra forward per layer, no autograd
graph, works for any architecture the hook can reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.autograd.forward_ad as fwAD
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "gpt2"
LN2 = math.log(2.0)


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    model_name = model_name.strip()
    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    # Eager attention: scaled_dot_product_attention's fused kernels don't all
    # support forward-mode AD, and the J-lens needs a JVP through every block.
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model = model.to(device).eval()
    # No reverse-mode graphs, ever: we only read activations and push tangents.
    model.requires_grad_(False)
    return model, tok, device


def _final_norm_and_unembed(model):
    """The defining approximation of the logit lens: reuse the *final* norm and
    the unembedding for every layer, even ones never normalized this way."""
    # GPT-2 / GPT-Neo family
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f, model.lm_head
    # Llama / Qwen / Mistral / Gemma family
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm, model.lm_head
    raise ValueError(
        f"Don't know the final-norm path for {type(model).__name__}; "
        "add it to _final_norm_and_unembed()."
    )


def _blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Don't know the block list for {type(model).__name__}.")


def _clean(s: str) -> str:
    s = s.replace("\n", "⏎").replace("\t", "⇥")
    return s if s != "" else "∅"


def _topk(probs: torch.Tensor, tok, k: int) -> list[dict]:
    p, i = probs.topk(k)
    return [{"t": _clean(tok.decode([int(ti)])), "p": float(pi)} for ti, pi in zip(i, p)]


def _jvp_transported(model, input_ids: torch.Tensor, block_input: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Transport rung ``layer_idx``'s last-position content into the final
    residual basis: J·h where J = ∂(blocks layer_idx..N-1)/∂rung.

    Mirrors the mini-GPT J-lens exactly: J covers only the residual→residual
    remainder — the transported vector is then decoded through the *real*
    (nonlinear) final norm + unembed. Linearizing the final LayerNorm too would
    strip the radial component that carries the prediction's confidence.

    One forward-mode AD pass: a pre-hook swaps the block's incoming hidden
    state for a dual tensor carrying tangent = its own last-position content;
    a second pre-hook on the final norm reads the tangent back off its input
    (= the last block's output, i.e. the pre-norm final residual).
    """
    tangent = torch.zeros_like(block_input)
    tangent[0, -1] = block_input[0, -1]

    def inject(module, args, kwargs):
        if args:
            dual = fwAD.make_dual(args[0], tangent)
            return (dual, *args[1:]), kwargs
        dual = fwAD.make_dual(kwargs["hidden_states"], tangent)
        kwargs = {**kwargs, "hidden_states": dual}
        return args, kwargs

    captured: list[torch.Tensor | None] = [None]

    def capture(module, args):
        _, tan = fwAD.unpack_dual(args[0])
        captured[0] = None if tan is None else tan[0, -1].detach().clone()

    final_norm, _ = _final_norm_and_unembed(model)
    h_inject = _blocks(model)[layer_idx].register_forward_pre_hook(inject, with_kwargs=True)
    h_capture = final_norm.register_forward_pre_hook(capture)
    try:
        # Forward-mode AD needs grad mode ON to thread tangents; with every
        # param at requires_grad=False this still builds no reverse graph.
        with torch.enable_grad(), fwAD.dual_level():
            model(input_ids)
    finally:
        h_inject.remove()
        h_capture.remove()
    if captured[0] is None:
        raise RuntimeError("forward-mode tangent did not propagate (op without JVP support?)")
    return captured[0]


@dataclass
class LensReport:
    tokens: list[str]
    layers: list[str]
    grid: list[list[list[dict]]]  # [layer][pos] -> top-k {t, p}
    bits: list[float]             # -log2 p_layer(final top-1) at last position
    jbits: list[float] | None
    jtop: list[list[dict]] | None  # [layer] -> top-k of the J-decode, last position
    pred: dict                     # the model's real next-token prediction
    uniform: float                 # log2(vocab) — the knows-nothing reference cost


def _preflight(model, input_ids) -> None:
    """Reject a prompt the model can't index BEFORE it reaches the GPU.

    A token id >= vocab, or a sequence longer than the positional table, would
    index out of range inside a CUDA embedding kernel -- which fires a
    device-side assert that poisons the whole process's CUDA context, so every
    later request (even loading another model) then fails. On CPU it's a plain
    IndexError; either way we'd rather fail clean here with a clear message.
    The tiny locally-trained models are where this bites: `local/add-*` has
    n_positions=16, so the web/TUI default prompt (~34 char-tokens) overflows it.
    """
    seq_len = input_ids.shape[1]
    max_pos = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_pos is not None and seq_len > max_pos:
        raise ValueError(
            f"prompt is {seq_len} tokens but this model only has {max_pos} "
            f"positions — shorten the prompt (this is a small model)."
        )
    vocab = model.get_input_embeddings().num_embeddings
    if seq_len and int(input_ids.max()) >= vocab:
        raise ValueError(f"prompt has a token id past this model's vocab of {vocab}.")


@torch.no_grad()
def lens_report(model, tok, text: str, top_k: int = 5, jlens: bool = False) -> LensReport:
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")  # on CPU; validate before touching the GPU
    input_ids = enc["input_ids"]
    _preflight(model, input_ids)
    enc = enc.to(device)
    input_ids = enc["input_ids"]

    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states  # n_layers+1 tensors [1, seq, d]; last one is post-final-norm
    final_norm, unembed = _final_norm_and_unembed(model)
    n = len(hidden)

    final_probs = out.logits[0, -1].float().softmax(-1)
    pred_id = int(final_probs.argmax())
    pred = {
        "token": _clean(tok.decode([pred_id])),
        "p": float(final_probs[pred_id]),
        "bits": -math.log(float(final_probs[pred_id])) / LN2,
    }

    grid: list[list[list[dict]]] = []
    bits: list[float] = []
    for idx, h in enumerate(hidden):
        # The classic gotcha: HF already ran the last hidden state through the
        # final norm — re-normalizing it would corrupt it. Norm only raw rungs.
        normed = h if idx == n - 1 else final_norm(h)
        probs = unembed(normed)[0].float().softmax(-1)  # [seq, vocab]
        grid.append([_topk(probs[p], tok, top_k) for p in range(probs.shape[0])])
        bits.append(-math.log(max(float(probs[-1, pred_id]), 1e-30)) / LN2)

    jbits: list[float] | None = None
    jtop: list[list[dict]] | None = None
    if jlens:
        try:
            jbits, jtop = [], []
            for idx in range(n):
                if idx == n - 1:
                    # Top rung: J = identity, the two lenses agree by construction.
                    jp = final_probs
                else:
                    jh = _jvp_transported(model, input_ids, hidden[idx], idx)
                    jp = unembed(final_norm(jh)).float().softmax(-1)
                jbits.append(-math.log(max(float(jp[pred_id]), 1e-30)) / LN2)
                jtop.append(_topk(jp, tok, top_k))
        except Exception as e:
            # The JVP path on some architectures tries to JIT a Triton kernel;
            # in a slim image with no C compiler that raises. Degrade to the
            # classic lens (jbits stays null) rather than failing the whole
            # request. Give the image a compiler to get the J-lens back.
            jbits, jtop = None, None
            print(f"[lens] J-lens unavailable ({type(e).__name__}: {e}); classic lens only")

    layer_names = ["embed"] + [f"layer {i}" for i in range(n - 2)] + ["final"]
    tokens = [_clean(tok.decode([int(t)])) for t in input_ids[0]]
    return LensReport(tokens=tokens, layers=layer_names, grid=grid, bits=bits,
                      jbits=jbits, jtop=jtop, pred=pred,
                      uniform=math.log2(out.logits.shape[-1]))
