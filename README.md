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
| `src/lib/morphology/morphology.ts` | Morphology lens: word-bounded, frequency-weighted morph lexicon |
| `src/lib/player.svelte.ts` | Playback (index into the trace) |
| `src/components/*` | CostPanel, CostChart, CandidatesTable, Controls (shared) + per-lens views |

A new domain = implement one `MdlProblem` adapter; the engine, player, and most
of the UI come for free. Three lenses share this spine today — grammar,
morphology, and a mini-GPT training trace.

## Morphology lens

Feed it a **word list with frequencies** (`walking 8`). Every word starts spelled
out as characters; greedy MDL repeatedly joins the morph pair whose merge saves
the most bits — but only *within* a word, and weighted by how often the word
occurs. Shared stems (`walk`, `talk`) and affixes (`ing`, `ed`, `un‑`) emerge
because they recur across the vocabulary and so repay their lexicon entry many
times over in the corpus. Same engine, player, cost panel, and candidate table
as the grammar lens — only the data shape (a weighted word set) and the
word-bounded counting differ.

This is the *agglomerative* variant (RePair/BPE over a weighted word list). It is
Morfessor in spirit — two-part MDL over a morph lexicon — but not in mechanism:
canonical Morfessor Baseline starts from whole words and recursively splits,
revisiting earlier cuts. That recursive-split engine is the **split** lens below.

## Morphology — phase 2 (Morfessor split)

The same word list, optimized from the **opposite extreme**. Every word starts as
a single whole-word morph (lexicon huge, corpus tiny — one token per word); the
algorithm then re-segments words to share structure, the mirror image of the
merge lens. Watching both converge on the same data from opposite ends is the
lesson.

This lens needs a **different engine**, and the reason is instructive — it's the
scaling caveat from the merge lens turned structural:

- **The move is a re-segmentation, scored against live global counts.** Re-analysing
  one word means pulling it out of the corpus and choosing the segmentation that
  minimizes the *total* cost given every other word as-is. Because the corpus code
  is `−log₂ p` over morph tokens, a word's best cut depends on the whole model — so
  the engine maintains incremental morph counts (`remove`/`add`/`reanalyse`) rather
  than recomputing a global cost per candidate.
- **It loops in epochs, not a single argmin.** It sweeps the word list, re-analysing
  each word, until a full pass changes nothing. Because "keep the current cut" is
  always a candidate, every step is non-increasing, so the trace is a clean
  descending staircase that still shows *re-analysis*: a word can flip its cut in a
  later epoch because other words moved the counts beneath it. That breaks the merge
  engine's "one global best move per step" invariant — hence a separate loop.
- **What stays free:** the `Step[]` trace, `Player`, transport, panels, cost panel,
  and evolution chart all apply unchanged. Only the search loop and the
  count-maintenance differ.

Words are short, so each re-analysis enumerates every segmentation and scores it by
the exact cost — correct by construction, no approximation (the classic recursive
`O(n²)` splitter is just an optimization we don't need at this scale).

## Roadmap

- **Graph domain** — SUBDUE-style substructure compression.
- **Speed (only if real vocabularies are loaded)** — the merge lens scores
  candidates by a full `cost(apply(...))` recompute; both code modes admit an exact
  `O(1)` delta (`L = T·log₂T − Σ f·log₂f` localizes a merge to a few frequencies).
  Worth adding behind a dev-time equality assertion against `cost()`; unnecessary
  for toy inputs.
