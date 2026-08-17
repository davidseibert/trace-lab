# trace·lab

An interactive, introspective sandbox for building deep intuition about
**description length** — how many bits it costs to describe your data, and why
the model that predicts best is the model that compresses best.

> **total = L(M) + L(D | M)** — the best model of your data is the one that lets
> you describe the model *and* the data-given-the-model in the fewest total bits.
> Too simple makes `L(D|M)` huge; too complex makes `L(M)` huge. MDL finds the
> trough under the chosen code: added complexity must repay its own bit cost.

Every lens turns some corner of that idea into a **falling bits curve** you can
scrub: a RePair grammar, morphology, a SUBDUE graph, an arithmetic coder, a
mini-GPT you can train and do fact-surgery on, and the **logit lens** over real
GPT-2 / Qwen — the same `−log₂ p` axis under all of them. *Prediction is
compression;* the lenses are that sentence, made tangible.

The web app also has **Math·foundations** (`#/math`): a linked teaching path
from binary choices through surprisal, entropy, arithmetic coding, and
cross-entropy. Its live exercises reuse the same probability and coder
implementations as the exhibits, so the derivation and the instrument cannot
quietly disagree.

## Two front-ends, one engine

The same lens payload drives two clients — pick your habitat:

- **web/** — a Svelte sandbox: every lens, with a shared transport (play, step,
  scrub) over an immutable trace. Runs standalone for everything except the
  real-model lenses. Hash-routed: `#/` is a card index grouped into three
  tiers — **instruments** (open-ended workbenches over real models, engine
  required), **toy models** (hand-sized neural nets), and **concepts** (closed
  theory exhibits) — and each lens keeps its settings (prompt, seed,
  temperature, toggles) in its URL, so a refresh keeps your state and a copied
  link is a reproduction recipe.
- **tui/** — an [OpenTUI](https://github.com/anomalyco/opentui) terminal
  front-end for the real-model logit lens / J-lens. A native-terminal grid of
  per-layer predictions, shaded by confidence.
- **engine/** — a FastAPI + torch service both talk to over HTTP. Models load
  once and stay resident, so switching models mid-session is free.

## Run

Everything is wrapped in a `Makefile` (`make help` lists it):

| command | what it does |
|---|---|
| `make up` | web + engine (GPU) → http://localhost:5180, engine on :5181 |
| `make up-cpu` | same on CPU (slower J-lens, no GPU needed) |
| `make web` | just the Svelte sandbox — every lens except the real-model one |
| `make tui` | the OpenTUI terminal front-end (brings the GPU engine up first) |
| `make smoke` | one-shot bits-ladder sanity check (`PROMPT="…"`, `MODEL=…` to override) |
| `make train` | train a 1.2M-param GPT-2 on 2-digit addition into the shared volume |
| `make build` / `make down` / `make clean` | build images / stop / also remove images |

Without Docker: `cd web && bun install && bun run dev` (sandbox on :5180);
`cd engine && uv sync && uv run uvicorn main:app --port 5181` (engine, needs a
working CUDA/PyTorch); `cd tui && bun install && bun run start` (terminal client;
`LENS_ENGINE` points it elsewhere).

## The lenses

Each is a way to write, read, or account for structure in bits. See
**[docs/lenses.md](docs/lenses.md)** for how each one works.

| lens | data | the move |
|---|---|---|
| **Grammar** | a string | name the digram that saves the most bits (RePair) |
| **Morph·merge** | a weighted word list | merge morphs within a word (BPE-style) |
| **Morph·split** | the same word list | re-segment from whole words (Morfessor) |
| **Graph·SUBDUE** | a labeled graph | collapse a recurring subgraph into one node |
| **Coder** | a probability stream | arithmetic-code it to literal bits |
| **Mini-GPT** | a training trace | implant facts, read the residual ladder, J-lens |
| **Attention Lab** | hand-sized, editable Q/K/V matrices | watch scaled dot-product attention compute, one clickable cell at a time |
| **Hopfield·retrieve** | stored glyph / random patterns | corrupt one, watch the modern Hopfield update snap it back — the update rule IS attention (Ramsauer et al. 2020) |
| **Tic·tac** | tic-tac-toe, solved | an arch × signal grid against the solved game: causal GPT vs board encoder vs MLP, trained on sampled games, soft solver targets, or distillation — minimax agreement, emergent D₄ equivariance, and L1-sparse circuits, all scrubbable |
| **Tic·arena** | the whole roster | head-to-head duels, a round-robin table, and probe-suite report cards — toys vs solvers vs (with the engine) Qwen3/Gemma read as next-token distributions over the digit tokens |
| **Logit·real** | GPT-2 / Qwen | the classic logit lens + J-lens over a real model, in bits |
| **Hopfield·heads** | every attention head of a real model | read each head as one-step Hopfield retrieval; sweep β and classify retrievers / poolers / mixers |
| **Reason·trace** | Qwen3-0.6B thinking traces | stream a reasoning trace live, per-token bits ladder as it's born — reasoning as compression |
| **Train·real** | the tiny addition GPT-2 | train it live on the engine (same run as `make train`); the checkpoints land in Logit·real's model picker |

## Architecture

A new domain = one `MdlProblem` adapter; the engine, player, and most of the UI
come for free.

| path | role |
|------|------|
| `web/src/lib/lenses.ts` | the lens registry — id, tier, blurb, cross-links; nav, router, and index all read it |
| `web/src/lib/router.svelte.ts` | hash router + settings-in-URL sync (`#/<lens>?…`) |
| `web/src/lib/guides.ts` | the ✓/✗ "How to read this" content every lens mounts |
| `web/src/lib/mdl/` | domain-agnostic MDL contracts + the greedy `trace()` runner |
| `web/src/lib/*/` | per-lens adapters (grammar, morphology, graph, coder, llm) |
| `web/src/lib/player.svelte.ts` | playback: an index into the immutable trace |
| `web/src/components/` | shared transport + cost/candidate panels, plus per-lens views |
| `engine/lens.py` | the real-model logit lens + J-lens (plain `transformers`) |
| `engine/main.py` | the FastAPI service (`/health`, `/lens`, `/next`, `/column`, SSE `/chat`, `/attn`, `/hopfield`, `/ablate`, SSE `/train`) |
| `tui/src/` | the OpenTUI front-end (talks to the engine over HTTP) |
| `tui/src/spectate.ts` | read-only sidecar in the TUI: `/state` + `/screen` on :5182 |
| `mcp/server.ts` | MCP bridge — lets Claude Code spectate the TUI + query the engine |

## Spectating from Claude Code (MCP)

`.mcp.json` registers a `trace-lab` MCP server (`mcp/server.ts`, runs under
bun — `cd mcp && bun install` once). While the TUI is running, Claude can see
exactly what you see and run its own experiments alongside:

| tool | what it does |
|---|---|
| `get_screen` | the TUI's current frame, as literal text |
| `get_state` | model/prompt/selection + the selected column's bits ladder |
| `get_grid` | the full layer × position lens grid (top-k per cell) |
| `query_lens` / `query_column` | run any prompt against the engine directly, without touching your view |
| `hopfield_heads` | every head read as one-step Hopfield retrieval — regime census across a β sweep |
| `engine_health` | device, available/resident models, sidecar reachability |

The TUI serves this through a read-only HTTP sidecar on **:5182**
(`tui/src/spectate.ts`; `TUI_SPECTATE_PORT=0` disables it). `make tui` passes
`--service-ports` so the container publishes it; the MCP server reads
`TUI_SPECTATE` / `LENS_ENGINE` to find the sidecar and engine elsewhere.

## Model weights

Weights live in the **`hf-cache` named Docker volume**, shared with
[x-logit-lens](../x-logit-lens) — both compose files pin `name: hf-cache`
(`external`, so compose can never delete it, not even `down -v`), so each model
downloads once across both projects and loads fast (the volume lives inside the
WSL2 VM). The Makefile creates it on first use; inspect it with
`docker run --rm -v hf-cache:/v alpine du -sh /v/hf`.

`WEB_PORT=5199 make up` remaps the web port if something squats on 5180. The
engine's host port must stay 5181 (the browser calls it directly) unless you
also set `VITE_ENGINE_URL`.
