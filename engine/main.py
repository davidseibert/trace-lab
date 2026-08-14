"""FastAPI engine service for the Logit·real lens.

Run from engine/:

    uv run uvicorn main:app --port 5181

Set HF_HOME first to reuse an existing HuggingFace cache (e.g. the one
x-logit-lens keeps under its data/hf). Models load lazily on first request and
stay resident; gpt2 / gpt2-medium / gpt2-large / Qwen2.5-0.5B all fit a 12 GB
card together.
"""
from __future__ import annotations

import gc
import os
import re
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lens import (ablate_report, attn_report, column_report, lens_report, load_model,
                  pick_device, reason_events, render_chat)

# The Qwen3-0.6B pair mirrors Raschka's *Build a Reasoning Model from Scratch*
# checkpoints: his base/reasoning .pth files are repackagings of these exact
# weights, so lensing these = lensing the book's models.
#
# Four architecture families are represented on purpose — the lens is only
# interesting if you can watch the SAME reading change shape across them:
#   GPT-2   (transformer.h / ln_f, fused c_attn)
#   Qwen    (model.layers / model.norm, split q/k/v)
#   Gemma 3 (same Llama-shaped paths; tied embeddings, sliding-window layers)
#   Llama   (Llama-3.2 and SmolLM2 are both LlamaForCausalLM, but with very
#            different vocabs — 128k vs 49k — so `uniform`, the knows-nothing
#            reference cost, differs by ~1.4 bits between them. Same shape,
#            different bits axis: a good control pair.)
#
# Two base/instruct pairs are here on purpose — Qwen3-0.6B-Base/0.6B and
# Llama-3.2-1B/1B-Instruct. Same weights before and after post-training, so the
# ladder difference between them IS the instruction tuning.
#
# DeepSeek-R1-Distill is the second model that actually emits a <think> block.
# Both it and Qwen3 have dedicated marker tokens (151648/9 and 151667/8), but
# R1's chat template PRE-OPENS <think> in the prompt, so the opening marker
# never appears in the generated stream at all — an "is this token <think>?"
# test leaves the region closed for the whole trace. TraceView.svelte therefore
# scans for the markers in text space and seeds the state from the prompt.
#
# meta-llama/* is licence-gated: it needs an accepted licence on the HF account
# AND an HF_TOKEN in the engine's environment (docker-compose passes one
# through). Without the token these 404/403 at load time, not at import.
#
# Two deliberate exclusions, both of which LOOK like obvious additions:
#   * Gemma 2 — final_logit_softcapping=30.0. The lens decodes rungs through
#     unembed(final_norm(h)), which bypasses the cap, while `pred` comes from
#     out.logits, which has it applied. The two would be measured on different
#     scales and the top rung would stop agreeing with the prediction by
#     construction. Gemma 3 dropped softcapping, which is why it's here instead.
#   * Phi-3 / Phi-4-mini — fused qkv_proj, so attn_report's v_proj hook fails
#     the same way GPT-2's does.
ALLOWED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

# `make tui MODEL=...` / `LENS_MODEL` can name a model that isn't on the list
# above; honour it as the default and add it, so trying a new architecture
# needs no code edit. The TUI reads `default` from /health to seed its dropdown.
DEFAULT = os.environ.get("LENS_MODEL", "").strip() or "gpt2"
if DEFAULT not in ALLOWED_MODELS:
    ALLOWED_MODELS.insert(0, DEFAULT)

# Locally-trained models (see train.py) live NEXT TO the HF cache in the shared
# volume — never inside hub/, which is a download cache keyed by repo revision.
# They're served as "local/<dirname>" and appear in /health's model list, which
# is what populates the UI dropdown.
LOCAL_DIR = Path(os.environ.get("LENS_LOCAL_DIR", "data/local"))
_LOCAL_RE = re.compile(r"^local/([A-Za-z0-9][A-Za-z0-9._-]*)$")


def local_models() -> list[str]:
    if not LOCAL_DIR.is_dir():
        return []
    return sorted(f"local/{p.name}" for p in LOCAL_DIR.iterdir() if (p / "config.json").is_file())


def _local_meta(dirname: str) -> dict:
    """Self-description for a local checkpoint: the note + sample prompts that
    train.py saved (lens_meta.json), plus the positional budget from
    config.json — so the UI can label the picker entry and swap in a prompt
    that actually fits instead of overflowing 16 positions with Eiffel."""
    d = LOCAL_DIR / dirname
    meta: dict = {}
    try:
        meta = json.loads((d / "lens_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if "n_positions" not in meta:
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            n_pos = cfg.get("n_positions") or cfg.get("max_position_embeddings")
            if n_pos:
                meta["n_positions"] = n_pos
        except (OSError, ValueError):
            pass
    return meta


def model_info() -> dict:
    """Per-model metadata for /health. `models` stays a flat list (the TUI
    reads it); richer clients merge this in by name."""
    info = {m: {"kind": "hub"} for m in ALLOWED_MODELS}
    for name in local_models():
        info[name] = {"kind": "local", **_local_meta(name.split("/", 1)[1])}
    return info


def _resolve(model_name: str) -> str:
    """Map an API model name to what from_pretrained should load."""
    if model_name in ALLOWED_MODELS:
        return model_name
    m = _LOCAL_RE.match(model_name)
    if m and (LOCAL_DIR / m.group(1) / "config.json").is_file():
        return str(LOCAL_DIR / m.group(1))
    raise HTTPException(
        400, f"unknown model {model_name!r}; allowed: {ALLOWED_MODELS + local_models()}"
    )

app = FastAPI(title="trace-lab engine", version="0.1.0")
# The Svelte dev server talks to us cross-origin; this service is loopback-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# App-wide error convention: ValueError = the caller's fault (bad prompt/pos/
# ids) → 400; RuntimeError = the engine's fault (CUDA, hooks, OOM) → 500.
# The SSE generators are the one exception — headers are already sent by the
# time they fail, so they emit an in-stream error event instead (see _sse).
@app.exception_handler(ValueError)
def _caller_fault(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
def _engine_fault(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Keep only a bounded set of models resident. Unbounded, every model you ever
# load stays on the GPU -- and once VRAM fills, Windows/WDDM spills into shared
# system memory and the whole machine hitches for seconds. So LRU-evict the
# least-recently-used model and hand its VRAM back before loading a new one.
# Measured peak VRAM, bf16, short prompt (RTX 4080 Laptop, 12 GB):
#   gemma-3-270m 0.85 · gemma-3-1b 2.5 · Llama-3.2-1B 2.8 · SmolLM2-1.7B 3.4 ·
#   Qwen3-1.7B 3.8 GiB
# So the default of 3 is fine for any mix that includes a small model, but THREE
# 1.7B-class models at once is ~11 GiB against ~10.5 GiB free once the desktop
# has its share — which is exactly the WDDM spill described above. Set
# LENS_MAX_RESIDENT=2 if you're hopping between the big ones. attn_report costs
# extra on top: eager attention materializes [heads, seq, seq] per layer.
MAX_RESIDENT = max(1, int(os.environ.get("LENS_MAX_RESIDENT", "3")))
_cache: "OrderedDict[str, tuple]" = OrderedDict()


def _free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get(model_name: str):
    if model_name in _cache:
        _cache.move_to_end(model_name)  # mark most-recently-used
        return _cache[model_name]
    # Free room BEFORE allocating the newcomer, so peak VRAM stays bounded.
    while len(_cache) >= MAX_RESIDENT:
        _, entry = _cache.popitem(last=False)  # least-recently-used
        del entry
        _free_vram()
    _cache[model_name] = load_model(_resolve(model_name))
    return _cache[model_name]


class LensRequest(BaseModel):
    model: str = DEFAULT
    prompt: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=20)
    jlens: bool = True
    # Greedily decode UP TO this many tokens server-side (stops at EOS) and
    # lens over prompt+continuation — a column per generated token. Sized for
    # reasoning traces, hence the generous cap.
    rollout: int = Field(default=0, ge=0, le=2048)
    # Wrap the prompt in the model's chat template before lensing. With
    # thinking=True a template that supports it (Qwen3) opens a <think> block,
    # so the rollout IS the reasoning trace.
    chat: bool = False
    thinking: bool = True


class ColumnRequest(BaseModel):
    model: str = DEFAULT
    prompt: str = Field(min_length=1, max_length=8000)
    # Column of the (rollout-extended) sequence to compute the ladder at.
    pos: int = Field(ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    jlens: bool = True
    rollout: int = Field(default=0, ge=0, le=2048)
    # Must match the /lens request's flags for the columns to line up.
    chat: bool = False
    thinking: bool = True
    # The exact token ids to lens (e.g. a streamed /chat trace). Preferred:
    # skips the prompt+rollout re-decode (no fork risk) and measures the
    # ladder against the token that actually follows pos — required for
    # sampled traces, where the taken token isn't the argmax. When set,
    # prompt/chat/thinking/rollout are ignored.
    ids: list[int] | None = Field(default=None, min_length=2, max_length=4096)


def _render(tok, req) -> str:  # any request with .chat / .prompt / .thinking
    """The text actually lensed: the raw prompt, or its chat-templated form."""
    if not req.chat:
        return req.prompt
    return render_chat(tok, req.prompt, thinking=req.thinking)


@app.get("/")
def root():
    return {"service": "trace-lab engine", "health": "/health", "docs": "/docs", "lens": "POST /lens"}


@app.get("/health")
def health():
    # local_models() is re-scanned per call, so freshly-trained checkpoints show
    # up on the next UI refresh. (A RE-trained checkpoint under an existing name
    # stays stale in _cache until the engine restarts.)
    return {
        "ok": True,
        "device": pick_device(),
        "loaded": list(_cache),
        "models": ALLOWED_MODELS + local_models(),
        "default": DEFAULT,
        "model_info": model_info(),
    }


@app.post("/lens")
def lens(req: LensRequest):
    model, tok, _device = _get(req.model)
    # ValueError here = a prompt the model can't index (too long / out of
    # vocab), caught on CPU by _preflight before it could poison CUDA.
    report = lens_report(
        model, tok, _render(tok, req), top_k=req.top_k, jlens=req.jlens, rollout=req.rollout
    )
    return {"model": req.model, "prompt": req.prompt, **asdict(report)}


class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen3-0.6B"
    prompt: str = Field(min_length=1, max_length=8000)
    # chat=False streams a raw continuation instead of a templated turn.
    chat: bool = True
    thinking: bool = True
    max_new: int = Field(default=512, ge=1, le=2048)
    # 0 = greedy. Thinking mode is meant to be sampled (Qwen recommends ~0.6);
    # the seed is generated if omitted and echoed in the meta event, so any
    # run can be replayed exactly. Reported bits always price the model's
    # true distribution — temperature only picks the path.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, ge=0)


def _sse(events):
    """data:-frame an event iterator as Server-Sent Events. Exceptions become
    an in-stream error event — by the time a generator fails the 200 and its
    headers are already on the wire, so the app-level handlers can't fire and
    SSE can't change status."""
    try:
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream a greedy decode as Server-Sent Events, one per token, each
    carrying that token's classic-lens ladder — the live feed the reasoning
    lens renders as the model thinks. J-lens drill-ins go through /column
    (same prompt + chat flags, rollout = tokens generated so far)."""
    model, tok, _device = _get(req.model)
    text = _render(tok, req)
    return StreamingResponse(
        _sse(reason_events(model, tok, text, max_new=req.max_new,
                           temperature=req.temperature, seed=req.seed)),
        media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class TrainRequest(BaseModel):
    epochs: int = Field(default=40, ge=1, le=200)


# One training run at a time — the checkpoints are shared mutable state.
_train_lock = threading.Lock()


@app.post("/train")
def train_endpoint(req: TrainRequest):
    """Stream the tiny-addition training run (train.py) as SSE — the Train·real
    lens's live feed. Checkpoints land in LOCAL_DIR and show up in /health on
    the next call; the model cache drops its local/ entries afterwards so a
    re-trained checkpoint is re-loaded, not served stale."""
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, "a training run is already in progress")

    def events():
        # Deferred import inside the generator, so an import failure surfaces
        # through _sse as an error event rather than a broken stream.
        from train import train_run

        yield from train_run(epochs=req.epochs, out=LOCAL_DIR)

    def gen():
        try:
            yield from _sse(events())
        finally:
            for k in list(_cache):
                if k.startswith("local/"):
                    _cache.pop(k, None)
            _free_vram()
            _train_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


class AttnRequest(BaseModel):
    model: str = "Qwen/Qwen3-0.6B"
    # The exact sequence (prompt ids + streamed ids) — same contract as /column.
    ids: list[int] = Field(min_length=2, max_length=4096)
    # Destination position: whose attention rows to read. Only ids[:pos+1] is
    # forwarded; eager attention materializes [heads, seq, seq] per layer, so
    # pos is capped to keep peak VRAM sane on a 12 GB card.
    pos: int = Field(ge=1, le=1500)
    # Optionally include one head's full value-weighted row.
    layer: int | None = Field(default=None, ge=0)
    head: int | None = Field(default=None, ge=0)


@app.post("/attn")
def attn(req: AttnRequest):
    """Every head's attention row at one destination token, plus per-head
    stats (entropy, sink mass, top sources) and raw/value-weighted aggregates."""
    model, _tok, _device = _get(req.model)
    return {"model": req.model,
            **attn_report(model, req.ids, req.pos,
                          pick_layer=req.layer, pick_head=req.head)}


class AblateRequest(BaseModel):
    model: str = "Qwen/Qwen3-0.6B"
    ids: list[int] = Field(min_length=2, max_length=4096)
    # The token to re-price (ids[pos]), and the source span [mask_start,
    # mask_end) that everything from mask_end onward is forbidden to read.
    pos: int = Field(ge=1)
    mask_start: int = Field(ge=0)
    mask_end: int = Field(ge=1)
    top_k: int = Field(default=5, ge=1, le=20)


@app.post("/ablate")
def ablate(req: AblateRequest):
    """Attention-as-bits: re-price a token with a source region masked out of
    the entire downstream computation. delta_bits is what reading that region
    actually bought."""
    model, tok, _device = _get(req.model)
    return {"model": req.model,
            **ablate_report(model, tok, req.ids, req.pos, req.mask_start,
                            req.mask_end, top_k=req.top_k)}


@app.post("/column")
def column(req: ColumnRequest):
    """The depth ladder (classic bits + J-lens) at one selected column.

    /lens carries the ladder only for the last column; the UIs call this
    lazily when another column is selected — one forward pass plus, with
    jlens, one JVP per rung.
    """
    model, tok, _device = _get(req.model)
    # ValueError here = a prompt the model can't index, or pos past the end —
    # caught on CPU by _preflight / the range check before it could poison CUDA.
    report = column_report(
        model, tok, "" if req.ids is not None else _render(tok, req), req.pos,
        top_k=req.top_k, jlens=req.jlens, rollout=req.rollout, ids=req.ids,
    )
    return {"model": req.model, "prompt": req.prompt, **asdict(report)}
