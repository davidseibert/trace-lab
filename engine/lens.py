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
# torch/math logs are natural (nats); bits = nats / ln 2, so every
# "-log(p) / LN2" below is -log2 p — the ideal code length of p in bits.
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


def render_chat(tok, text: str, thinking: bool = True) -> str:
    """Wrap a user message in the model's chat template, generation-ready.

    ``enable_thinking`` is honoured by templates that know it (Qwen3's opens a
    <think> block); templates that don't simply ignore the extra variable.
    """
    if tok.chat_template is None:
        raise ValueError(
            "this model's tokenizer has no chat template — use an instruct/"
            "reasoning variant, or turn chat mode off."
        )
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def _decode_rollout(model, input_ids: torch.Tensor, rollout: int) -> torch.Tensor:
    """Greedily decode up to ``rollout`` tokens onto ``input_ids``, KV-cached.

    ``model.generate`` instead of a re-forward-per-token loop: a reasoning
    trace is hundreds of tokens, and without the cache that loop is O(n²) in
    forwards. Stops early at EOS, so ``rollout`` is a budget, not a promise —
    callers must measure the returned sequence, not assume prompt+rollout.
    """
    if not rollout:
        return input_ids
    eos = model.generation_config.eos_token_id
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    return model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=rollout,
        do_sample=False,
        pad_token_id=eos,  # silences the pad=eos warning; nothing is padded
    )


def _clean(s: str) -> str:
    s = s.replace("\n", "⏎").replace("\t", "⇥")
    return s if s != "" else "∅"


def _topk(probs: torch.Tensor, tok, k: int) -> list[dict]:
    p, i = probs.topk(k)
    return [{"t": _clean(tok.decode([int(ti)])), "p": float(pi)} for ti, pi in zip(i, p)]


def _jvp_transported(
    model, input_ids: torch.Tensor, block_input: torch.Tensor, layer_idx: int, pos: int = -1
) -> torch.Tensor:
    """Transport rung ``layer_idx``'s content at position ``pos`` into the final
    residual basis: J·h where J = ∂(blocks layer_idx..N-1)/∂rung.

    Mirrors the mini-GPT J-lens exactly: J covers only the residual→residual
    remainder — the transported vector is then decoded through the *real*
    (nonlinear) final norm + unembed. Linearizing the final LayerNorm too would
    strip the radial component that carries the prediction's confidence.

    One forward-mode AD pass: a pre-hook swaps the block's incoming hidden
    state for a dual tensor carrying tangent = the content at ``pos``; a second
    pre-hook on the final norm reads the tangent back off its input at ``pos``
    (= the last block's output, i.e. the pre-norm final residual). One seed
    position per pass is structural: JVPs are linear in the tangent, so seeding
    several positions at once would sum their transports through attention.
    """
    # A JVP is a directional derivative: seeding tangent v yields J·v in one
    # pass, no Jacobian ever materialized. Here v = the rung's own content at
    # pos (zeros elsewhere), so the tangent that arrives at the top is exactly
    # J·h — this vector as the rest of the network reshapes it, to first order.
    tangent = torch.zeros_like(block_input)  # [1, seq, d]
    tangent[0, pos] = block_input[0, pos]

    def inject(module, args, kwargs):
        # A dual tensor packs (primal, tangent); every op inside dual_level()
        # then propagates the tangent by its own JVP rule. hidden_states
        # arrives positionally (GPT-2 family) or as a kwarg (Llama family).
        if args:
            dual = fwAD.make_dual(args[0], tangent)
            return (dual, *args[1:]), kwargs
        dual = fwAD.make_dual(kwargs["hidden_states"], tangent)
        kwargs = {**kwargs, "hidden_states": dual}
        return args, kwargs

    captured: list[torch.Tensor | None] = [None]

    def capture(module, args):
        # Read at the final norm's INPUT — the pre-norm final residual — so J
        # spans exactly blocks layer_idx..N-1 and nothing else.
        _, tan = fwAD.unpack_dual(args[0])
        captured[0] = None if tan is None else tan[0, pos].detach().clone()  # [d]

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
    n_prompt: int                  # tokens[:n_prompt] are the prompt; the rest are rollout


@dataclass
class ColumnReport:
    """The depth ladder at one position — what /lens reports for the last
    column, recomputed for any column the UI selects."""
    pos: int
    pred: dict                     # the model's real prediction *after* this column
    bits: list[float]
    jbits: list[float] | None
    jtop: list[list[dict]] | None
    uniform: float


def _ladder(model, tok, input_ids, out, hidden, pos: int, top_k: int, jlens: bool,
            target_id: int | None = None):
    """Classic bits per rung at ``pos`` — and, if asked, the J-lens decode per
    rung — measured against the model's top-1 prediction at ``pos`` (the token
    after that column), or against ``target_id`` when the caller knows which
    token was actually taken (a sampled trace, where taken ≠ argmax)."""
    final_norm, unembed = _final_norm_and_unembed(model)
    n = len(hidden)

    # logits: [1, seq, V] -> the row at pos: [V]. .float() upcasts from
    # bf16/fp16 BEFORE softmax so tiny tail probabilities (= large bit costs)
    # survive; softmax is max-subtracted internally, so overflow isn't the
    # worry — quantization of the tail is.
    final_probs = out.logits[0, pos].float().softmax(-1)
    pred_id = int(final_probs.argmax()) if target_id is None else target_id
    pred = {
        "token": _clean(tok.decode([pred_id])),
        "p": float(final_probs[pred_id]),
        "bits": -math.log(float(final_probs[pred_id])) / LN2,  # surprisal of the pick itself
    }

    # One entry per rung: decode rung idx's residual through the FINAL
    # norm+unembed (the lens approximation) and price pred_id under it. The
    # 1e-30 floor caps a rung's reportable cost at ~99.7 bits instead of
    # letting log(0) blow up when an early rung gives the token no mass.
    bits: list[float] = []
    for idx, h in enumerate(hidden):
        # Same gotcha as the grid: the last hidden state is already normed.
        normed = h[0, pos] if idx == n - 1 else final_norm(h[0, pos])  # [d]
        probs = unembed(normed).float().softmax(-1)  # [V]
        bits.append(-math.log(max(float(probs[pred_id]), 1e-30)) / LN2)

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
                    jh = _jvp_transported(model, input_ids, hidden[idx], idx, pos)
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

    return pred, bits, jbits, jtop


def _preflight(model, input_ids, rollout: int = 0) -> None:
    """Reject a prompt the model can't index BEFORE it reaches the GPU.

    A token id >= vocab, or a sequence longer than the positional table, would
    index out of range inside a CUDA embedding kernel -- which fires a
    device-side assert that poisons the whole process's CUDA context, so every
    later request (even loading another model) then fails. On CPU it's a plain
    IndexError; either way we'd rather fail clean here with a clear message.
    The tiny locally-trained models are where this bites: `local/add-*` has
    n_positions=16, so the web/TUI default prompt (~34 char-tokens) overflows it.
    Rollout tokens occupy positions too, so they count against the same budget.
    """
    seq_len = input_ids.shape[1]
    max_pos = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_pos is not None and seq_len + rollout > max_pos:
        grown = f" + {rollout} rollout tokens" if rollout else ""
        raise ValueError(
            f"prompt is {seq_len} tokens{grown} but this model only has {max_pos} "
            f"positions — shorten the prompt (this is a small model)."
        )
    vocab = model.get_input_embeddings().num_embeddings
    if seq_len and int(input_ids.max()) >= vocab:
        raise ValueError(f"prompt has a token id past this model's vocab of {vocab}.")


@torch.no_grad()
def lens_report(
    model, tok, text: str, top_k: int = 5, jlens: bool = False, rollout: int = 0
) -> LensReport:
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")  # on CPU; validate before touching the GPU
    input_ids = enc["input_ids"]
    _preflight(model, input_ids, rollout)
    enc = enc.to(device)
    input_ids = enc["input_ids"]
    n_prompt = input_ids.shape[1]

    # Rollout: greedily decode up to `rollout` tokens and lens over
    # prompt+continuation. Each grid column only attends to positions before it
    # (causal), so forcing the model's own answer in doesn't leak anything
    # backwards — it just gives every generated token its own column to watch
    # form with depth.
    input_ids = _decode_rollout(model, input_ids, rollout)
    if rollout:
        enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states  # n_layers+1 tensors [1, seq, d]; last one is post-final-norm
    final_norm, unembed = _final_norm_and_unembed(model)
    n = len(hidden)

    grid: list[list[list[dict]]] = []
    for idx, h in enumerate(hidden):
        # The classic gotcha: HF already ran the last hidden state through the
        # final norm — re-normalizing it would corrupt it. Norm only raw rungs.
        normed = h if idx == n - 1 else final_norm(h)
        probs = unembed(normed)[0].float().softmax(-1)  # [seq, vocab]
        grid.append([_topk(probs[p], tok, top_k) for p in range(probs.shape[0])])

    pred, bits, jbits, jtop = _ladder(model, tok, input_ids, out, hidden, -1, top_k, jlens)

    layer_names = ["embed"] + [f"layer {i}" for i in range(n - 2)] + ["final"]
    tokens = [_clean(tok.decode([int(t)])) for t in input_ids[0]]
    return LensReport(tokens=tokens, layers=layer_names, grid=grid, bits=bits,
                      jbits=jbits, jtop=jtop, pred=pred,
                      uniform=math.log2(out.logits.shape[-1]), n_prompt=n_prompt)


def _kv_heads(model) -> int:
    return getattr(model.config, "num_key_value_heads", None) or model.config.num_attention_heads


# How many top source positions each head reports. Enough to see where a head
# is really looking without shipping the full row for every layer×head.
TOP_SOURCES = 8


@torch.no_grad()
def attn_report(model, ids: list[int], pos: int,
                pick_layer: int | None = None, pick_head: int | None = None) -> dict:
    """Every head's attention row at destination ``pos``: where that token
    looked while being computed.

    Requires the eager attention path (which load_model already forces for the
    J-lens). Forwards only ``ids[:pos+1]`` — the rows at ``pos`` don't depend
    on anything after it. Returns per-head stats plus two aggregates over all
    layers×heads:

      - ``agg``  — mean raw attention received per source position
      - ``vagg`` — the same, value-weighted (a·‖v‖, renormalized): a head can
        stare at a token whose value vector is ~zero (attention sinks); this
        discounts those no-op looks.

    ``pick_layer``/``pick_head`` additionally return that one head's full
    value-weighted row for UI overlays. Memory note: eager attention
    materializes [heads, seq, seq] per layer, so seq is capped upstream.
    """
    input_ids = torch.tensor([ids], dtype=torch.long)
    _preflight(model, input_ids, 0)
    if not 0 < pos < input_ids.shape[1]:
        raise ValueError(f"pos {pos} out of range (need 0 < pos < {input_ids.shape[1]}).")
    device = next(model.parameters()).device
    input_ids = input_ids[:, : pos + 1].to(device)
    seq = pos + 1

    # ‖v‖ per source, per KV head, per layer — captured off each v_proj.
    vnorms: list[torch.Tensor] = []
    n_kv = _kv_heads(model)

    def grab_v(module, args, output):
        # v_proj output: [1, seq, kv_heads*head_dim]; [0] drops batch, the
        # view splits the fused head dimension back out.
        v = output[0].view(seq, n_kv, -1)  # [seq, kv_heads, head_dim]
        vnorms.append(v.float().norm(dim=-1).T.cpu())  # [kv_heads, seq]

    hooks = [blk.self_attn.v_proj.register_forward_hook(grab_v) for blk in _blocks(model)]
    try:
        out = model(input_ids=input_ids, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    n_heads = out.attentions[0].shape[1]
    group = n_heads // n_kv  # GQA: query head h shares KV head h // group
    agg = torch.zeros(seq)
    vagg = torch.zeros(seq)
    heads = []
    picked = None
    for L, att in enumerate(out.attentions):
        # att: [1, heads, dst, src] post-softmax — fix dst=pos to get each
        # head's row over sources (sums to 1 per head).
        rows = att[0, :, pos, :].float().cpu()  # [heads, seq]
        for h in range(n_heads):
            a = rows[h]
            # Value-weighting: scale each source's weight by ‖v‖, renormalize
            # back to a distribution. clamp_min guards 0/0 when every attended
            # value is ~zero (a pure-sink row).
            w = a * vnorms[L][h // group]
            w = w / w.sum().clamp_min(1e-9)
            agg += a
            vagg += w
            # Shannon entropy of the row, in bits: H = -Σ a·log2 a. The clamp
            # makes 0·log 0 contribute 0 instead of NaN. Low H = a focused
            # head; H near log2(seq) = uniform averaging.
            ent = float(-(a.clamp_min(1e-12) * a.clamp_min(1e-12).log2()).sum())
            tk = a.topk(min(TOP_SOURCES, seq))
            heads.append({
                "layer": L, "head": h,
                "entropy": round(ent, 3),
                "sink": round(float(a[0]), 4),  # mass parked on token 0 — the classic attention sink
                "top": [{"pos": int(p), "w": round(float(x), 4), "vw": round(float(w[p]), 4)}
                        for x, p in zip(tk.values, tk.indices)],
            })
            if L == pick_layer and h == pick_head:
                picked = {"vrow": [round(float(x), 5) for x in w]}
    n_total = len(out.attentions) * n_heads
    return {
        "pos": pos, "seq": seq,
        "n_layers": len(out.attentions), "n_heads": n_heads,
        "agg": [round(float(x), 5) for x in agg / n_total],   # mean over layers×heads
        "vagg": [round(float(x), 5) for x in vagg / n_total],
        "heads": heads,
        "picked": picked,
    }


@torch.no_grad()
def ablate_report(model, tok, ids: list[int], pos: int, mask_start: int, mask_end: int,
                  top_k: int = 5) -> dict:
    """Re-price token ``ids[pos]`` with attention to sources
    [mask_start, mask_end) blocked for every position from mask_end on — the
    causal counterfactual "what would this token cost if the whole computation
    after the region couldn't read it". Attention-as-bits: the delta is what
    attending to that region actually bought.
    """
    input_ids = torch.tensor([ids], dtype=torch.long)
    _preflight(model, input_ids, 0)
    n = input_ids.shape[1]
    if not 0 < pos < n:
        raise ValueError(f"pos {pos} out of range (need 0 < pos < {n}).")
    if not 0 <= mask_start < mask_end <= pos:
        raise ValueError(f"need 0 <= mask_start < mask_end <= pos, got [{mask_start}, {mask_end}) vs pos {pos}.")
    device = next(model.parameters()).device
    prefix = input_ids[:, :pos].to(device)  # logits at pos-1 price ids[pos]
    target = int(ids[pos])

    def price(attention_mask=None):
        out = model(input_ids=prefix, attention_mask=attention_mask)
        probs = out.logits[0, -1].float().softmax(-1)
        return probs

    base = price()
    # HF attention masks are ADDITIVE, applied to scores before softmax:
    # 0 keeps a link, finfo.min drives its post-softmax weight to ~0. m[i, j]
    # governs destination i's view of source j. Handing the model an explicit
    # mask replaces its own, so triu(1) must rebuild the causal triangle
    # (i can't see j > i) before the block rule is stamped on top.
    dtype = next(model.parameters()).dtype
    neg = torch.finfo(dtype).min
    m = torch.zeros(pos, pos, dtype=dtype)
    m.masked_fill_(torch.ones(pos, pos, dtype=torch.bool).triu(1), neg)  # causal
    m[mask_end:, mask_start:mask_end] = neg  # rows after the region can't read it
    masked = price(m[None, None].to(device))  # [1, 1, dst, src]: broadcast over batch and heads

    def entry(probs):
        p = float(probs[target])
        return {"p": p, "bits": -math.log(max(p, 1e-30)) / LN2}

    b, mk = entry(base), entry(masked)
    return {
        "pos": pos, "token": _clean(tok.decode([target])),
        "mask": [mask_start, mask_end],
        "baseline": {**b, "top": _topk(base, tok, top_k)},
        "masked": {**mk, "top": _topk(masked, tok, top_k)},
        "delta_bits": mk["bits"] - b["bits"],
    }


@torch.no_grad()
def next_report(model, tok, text: str, top_k: int = 20) -> dict:
    """The next-token distribution at the end of a prompt — top-k only, no
    grid, no ladder. One forward pass; the lens machinery is skipped
    entirely. Exists because /lens spends ~99% of its wall time on the full
    layers×positions grid (measured 7.5 s vs 0.1 s for the same forward on
    Qwen3-0.6B): callers that want ONE distribution — Tic·arena's LLM
    players, any constrained-choice read — should not pay for a grid they
    never look at.
    """
    input_ids = tok(text, return_tensors="pt").input_ids
    _preflight(model, input_ids, 0)
    device = next(model.parameters()).device
    out = model(input_ids=input_ids.to(device))
    probs = out.logits[0, -1].float().softmax(-1)
    return {"n_tokens": int(input_ids.shape[1]), "top": _topk(probs, tok, top_k)}


# Regime thresholds — MUST match the toy lens's exported constants in
# web/src/lib/hopfield/hopfield.ts (RETRIEVAL_EFFK / GLOBAL_EFFK_FRAC), so the
# toy and the instrument speak one vocabulary.
HOPFIELD_RETRIEVAL_EFFK = 1.5
HOPFIELD_GLOBAL_EFFK_FRAC = 0.75


def _hopfield_regime(eff_k: float, n: int) -> str:
    if eff_k <= HOPFIELD_RETRIEVAL_EFFK:
        return "retrieval"
    if eff_k >= HOPFIELD_GLOBAL_EFFK_FRAC * n:
        return "global"
    return "metastable"


@torch.no_grad()
def hopfield_report(model, tok, text: str, pos: int, gammas: list[float]) -> dict:
    """Every attention head read as one-step modern-Hopfield retrieval
    (Ramsauer et al. 2020): the head's row at destination ``pos`` is
    softmax(β·scores) over the stored patterns (= earlier positions), and
    rescaling the inverse temperature by γ needs no pre-softmax logits at all —
    softmax(γ·z) = wᵞ / Σ wᵞ from the post-softmax row w. γ = 1 is the model's
    own β = 1/√d_k; the sweep asks "which retrieval regime is this head in,
    and how close to a phase boundary does the trained model sit?".

    Per head per γ: row entropy in bits, max weight, and the effective number
    of mixed patterns 2^H, classified into retrieval / metastable / global with
    the same thresholds as the Hopfield·retrieve toy lens.
    """
    for g in gammas:
        if not 0 < g <= 64:
            raise ValueError(f"gammas must be in (0, 64], got {g}.")
    input_ids = tok(text, return_tensors="pt").input_ids
    _preflight(model, input_ids, 0)
    n = input_ids.shape[1]
    if pos < 0:
        pos = n + pos
    if not 0 < pos < n:
        raise ValueError(f"pos {pos} out of range (need 0 < pos < {n}).")
    device = next(model.parameters()).device
    input_ids = input_ids[:, : pos + 1].to(device)
    seq = pos + 1

    out = model(input_ids=input_ids, output_attentions=True)

    heads = []
    for L, att in enumerate(out.attentions):
        rows = att[0, :, pos, :].float().cpu()  # [heads, seq] post-softmax
        for h in range(rows.shape[0]):
            a = rows[h].clamp_min(1e-30)
            curves = []
            regime_at_1 = None
            for g in gammas:
                # wᵞ renormalized — exactly softmax at inverse temperature γ·β.
                w = a**g
                w = w / w.sum()
                ent = float(-(w * w.clamp_min(1e-12).log2()).sum())
                eff = 2.0**ent
                regime = _hopfield_regime(eff, seq)
                curves.append({
                    "gamma": g,
                    "entropy_bits": round(ent, 3),
                    "max_w": round(float(w.max()), 4),
                    "eff_k": round(eff, 2),
                    "regime": regime,
                })
                if g == 1.0:
                    regime_at_1 = regime
            heads.append({"layer": L, "head": h, "curves": curves,
                          "regime_at_1": regime_at_1})

    return {
        "pos": pos, "seq": seq,
        "n_layers": len(out.attentions), "n_heads": out.attentions[0].shape[1],
        "gammas": gammas,
        "tokens": [_clean(tok.decode([int(t)])) for t in input_ids[0]],
        "heads": heads,
    }


@torch.no_grad()
def reason_events(model, tok, text: str, max_new: int = 512,
                  temperature: float = 0.0, seed: int | None = None):
    """Stream a decode as lens events: one dict per generated token, carrying
    the classic-lens ladder of the column that produced it.

    ``temperature`` > 0 samples instead of taking the argmax (Qwen3's thinking
    mode is meant to be sampled — greedy ruminates). Sampling is seeded and the
    seed is reported in the meta event, so a run can be replayed exactly. The
    reported ``p``/``bits`` are ALWAYS the model's true (unscaled) probability
    of the emitted token — temperature picks the path, never the price.

    The ladder is nearly free during decode — the incremental forward already
    has every rung's hidden state for the newest position, so each rung costs
    one unembed matmul (batched across rungs below). The J-lens is NOT
    streamed: at a JVP per rung per token it would dominate generation;
    clients drill into a chosen column with /column instead.

    Events:
      {"event": "meta", tokens, ids, layers, uniform, n_prompt, temperature, seed}
      {"event": "tok", pos, id, t, p, bits, rtop}   # pos: index in full sequence
      {"event": "done", reason: "eos" | "budget"}

    ``bits[r]`` is −log₂ p_rung(emitted token). ``rtop[r]`` is each rung's own
    top-1, the streamed grid column. ``ids`` + per-token ``id`` let clients
    hand the exact sequence back to /column for drill-ins — no re-decode.
    """
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")  # on CPU; validate before touching the GPU
    input_ids = enc["input_ids"]
    _preflight(model, input_ids, max_new)
    input_ids = input_ids.to(device)
    n_prompt = input_ids.shape[1]
    final_norm, unembed = _final_norm_and_unembed(model)

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos} if eos is not None else set()

    n_blocks = len(_blocks(model))
    layers = ["embed"] + [f"layer {i}" for i in range(n_blocks - 1)] + ["final"]
    vocab = model.get_input_embeddings().num_embeddings

    sampler = None
    if temperature > 0:
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        sampler = torch.Generator()  # CPU generator: device-independent replay
        sampler.manual_seed(seed)

    yield {
        "event": "meta",
        "tokens": [_clean(tok.decode([int(t)])) for t in input_ids[0]],
        "ids": input_ids[0].tolist(),
        "layers": layers,
        "uniform": math.log2(vocab),
        "n_prompt": n_prompt,
        "temperature": temperature,
        "seed": seed,
    }

    past = None
    cur = input_ids
    for step in range(max_new):
        out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        probs = out.logits[0, -1].float().softmax(-1)  # the TRUE distribution: prices come from here
        if sampler is None:
            next_id = int(probs.argmax())
        else:
            # Temperature rescales logits before softmax: T<1 sharpens toward
            # the argmax, T>1 flattens toward uniform. Only the DRAW uses this
            # distribution; the reported p/bits stay on `probs` above.
            temp_probs = (out.logits[0, -1].float() / temperature).softmax(-1)
            next_id = int(torch.multinomial(temp_probs.cpu(), 1, generator=sampler).item())

        # One decode for every rung at once: stack the newest position's hidden
        # state per rung, final-norm all but the already-normed last one.
        hs = torch.stack([h[0, -1] for h in out.hidden_states])  # [n_rungs, d]
        normed = torch.cat([final_norm(hs[:-1]), hs[-1:]])
        rung_probs = unembed(normed).float().softmax(-1)  # [n_rungs, vocab]
        tgt = rung_probs[:, next_id].clamp_min(1e-30)  # each rung's p(emitted token)
        bits = (-tgt.log() / LN2).tolist()  # -log2 p across all rungs in one shot
        rtop = [_clean(tok.decode([int(i)])) for i in rung_probs.argmax(-1)]

        yield {
            "event": "tok",
            "pos": n_prompt + step,
            "id": next_id,
            "t": _clean(tok.decode([next_id])),
            "p": float(probs[next_id]),
            "bits": [round(b, 3) for b in bits],
            "rtop": rtop,
        }
        if next_id in eos_ids:
            yield {"event": "done", "reason": "eos"}
            return
        # Incremental decode: forward only the new token; past_key_values
        # carries every earlier position's keys/values.
        cur = torch.tensor([[next_id]], device=device)
    yield {"event": "done", "reason": "budget"}


@torch.no_grad()
def column_report(
    model, tok, text: str, pos: int, top_k: int = 5, jlens: bool = True, rollout: int = 0,
    ids: list[int] | None = None,
) -> ColumnReport:
    """The depth ladder at column ``pos`` of the (rollout-extended) sequence.

    /lens only carries the ladder for the last column — this recomputes it for
    any column the UI selects, at the cost of one forward pass plus (with
    jlens) one JVP per rung.

    ``ids`` is the preferred path: the caller hands over the exact token
    sequence (e.g. a streamed /chat trace) and the ladder is measured against
    the token that actually follows ``pos`` — correct even for sampled traces,
    and immune to re-decode forks at near-ties. Without ``ids``, the rollout
    is re-decoded greedily from the prompt, which reproduces the sequence only
    for greedy traces (and even then bf16 near-ties can occasionally fork).
    """
    device = next(model.parameters()).device
    if ids is not None:
        input_ids = torch.tensor([ids], dtype=torch.long)  # validate on CPU first
        _preflight(model, input_ids, 0)
        input_ids = input_ids.to(device)
    else:
        enc = tok(text, return_tensors="pt")  # on CPU; validate before touching the GPU
        input_ids = enc["input_ids"]
        _preflight(model, input_ids, rollout)
        input_ids = input_ids.to(device)
        input_ids = _decode_rollout(model, input_ids, rollout)

    seq_len = input_ids.shape[1]
    if not 0 <= pos < seq_len:
        raise ValueError(f"pos {pos} is out of range for {seq_len} tokens.")

    # With the sequence in hand, the followed token is known — measure that.
    target_id = int(input_ids[0, pos + 1]) if ids is not None and pos + 1 < seq_len else None

    out = model(input_ids=input_ids, output_hidden_states=True)
    pred, bits, jbits, jtop = _ladder(
        model, tok, input_ids, out, out.hidden_states, pos, top_k, jlens, target_id=target_id
    )
    return ColumnReport(pos=pos, pred=pred, bits=bits, jbits=jbits, jtop=jtop,
                        uniform=math.log2(out.logits.shape[-1]))
