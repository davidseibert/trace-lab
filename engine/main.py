"""FastAPI engine service for the Logit·real lens.

Run from engine/:

    uv run uvicorn main:app --port 5181

Set HF_HOME first to reuse an existing HuggingFace cache (e.g. the one
x-logit-lens keeps under its data/hf). Models load lazily on first request and
stay resident; gpt2 / gpt2-medium / gpt2-large / Qwen2.5-0.5B all fit a 12 GB
card together.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lens import lens_report, load_model, pick_device

ALLOWED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-0.5B"]

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
    if model_name not in _cache:
        _cache[model_name] = load_model(_resolve(model_name))
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
    # local_models() is re-scanned per call, so freshly-trained checkpoints show
    # up on the next UI refresh. (A RE-trained checkpoint under an existing name
    # stays stale in _cache until the engine restarts.)
    return {
        "ok": True,
        "device": pick_device(),
        "loaded": list(_cache),
        "models": ALLOWED_MODELS + local_models(),
    }


@app.post("/lens")
def lens(req: LensRequest):
    model, tok, _device = _get(req.model)
    try:
        report = lens_report(model, tok, req.prompt, top_k=req.top_k, jlens=req.jlens)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"model": req.model, "prompt": req.prompt, **asdict(report)}
