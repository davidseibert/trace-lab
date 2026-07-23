"""FastAPI engine service for the Logit·real lens.

Run from engine/:

    uv run uvicorn main:app --port 5181

Set HF_HOME first to reuse an existing HuggingFace cache (e.g. the one
x-logit-lens keeps under its data/hf). Models load lazily on first request and
stay resident; gpt2 / gpt2-medium / gpt2-large / Qwen2.5-0.5B all fit a 12 GB
card together.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lens import lens_report, load_model, pick_device

ALLOWED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-0.5B"]

app = FastAPI(title="x-mdl engine", version="0.1.0")
# The Svelte dev server talks to us cross-origin; this service is loopback-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, tuple] = {}


def _get(model_name: str):
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(400, f"unknown model {model_name!r}; allowed: {ALLOWED_MODELS}")
    if model_name not in _cache:
        _cache[model_name] = load_model(model_name)
    return _cache[model_name]


class LensRequest(BaseModel):
    model: str = "gpt2"
    prompt: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    jlens: bool = True


@app.get("/")
def root():
    return {"service": "x-mdl engine", "health": "/health", "docs": "/docs", "lens": "POST /lens"}


@app.get("/health")
def health():
    return {"ok": True, "device": pick_device(), "loaded": list(_cache), "models": ALLOWED_MODELS}


@app.post("/lens")
def lens(req: LensRequest):
    model, tok, _device = _get(req.model)
    try:
        report = lens_report(model, tok, req.prompt, top_k=req.top_k, jlens=req.jlens)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"model": req.model, "prompt": req.prompt, **asdict(report)}
