# MDL Explorer

An interactive, introspective sandbox for building deep intuition about the
**Minimum Description Length** principle.

> **total = L(M) + L(D | M)** — the best model of your data is the one that lets
> you describe the model *and* the data-given-the-model in the fewest total bits.
> A model too simple makes `L(D|M)` huge; too complex makes `L(M)` huge. MDL
> finds the trough — and never overfits, because complexity literally costs bits.

## Run

```bash
bun install
bun run dev      # http://localhost:5180
```

`bun run check` type-checks; `bun run build` produces a production bundle.

## What it does (string grammar lens)

Feed it a string. It runs a RePair-style greedy search: repeatedly name the
adjacent digram whose abbreviation removes the most bits, replace every
occurrence with a new rule symbol, and watch longer structures (`"the "`) emerge
as a hierarchy of rules.

The whole search is computed up front as an immutable trace; the UI is a pure
function of the current step. So you get a **debugger**:

- ▶ play / pause, step forward/back, scrub, and slow-motion speeds
- live **L(M) vs L(D|M)** breakdown with a shrinking stacked bar
- the description-length **evolution chart** (click to seek)
- the **candidate table**, ranked by Δbits — and it flags when MDL's pick differs
  from the merely most-frequent digram (frequency ≠ best compression)
- a **uniform log₂(V)** vs **Shannon −log₂(p)** code toggle, and a
  model-of-model **overhead** toggle — both change the decisions MDL makes

## Architecture

| Path | Role |
|------|------|
| `src/lib/mdl/types.ts` | Domain-agnostic MDL contracts (`MdlProblem`, `Step`, `CostBreakdown`) |
| `src/lib/mdl/engine.ts` | `trace()` — greedy search → immutable snapshot list |
| `src/lib/string/grammar.ts` | String lens: digram grammar, uniform + Shannon costs |
| `src/lib/player.svelte.ts` | Playback (index into the trace) |
| `src/components/*` | StreamView, CostPanel, CostChart, CandidatesTable, Controls |

A new domain = implement one `MdlProblem` adapter; the engine, player, and most
of the UI come for free.

## Roadmap

- **Morphology lens** — Morfessor-style word-list segmentation on the same engine.
- **Graph domain** — SUBDUE-style substructure compression.
