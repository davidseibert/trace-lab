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
| `src/lib/graph/graph.ts` | Graph lens: SUBDUE substructure compression (`canon.ts`, `match.ts`, `layout.ts`) |
| `src/lib/coder/arithmetic.ts` | Coder lens: model-agnostic arithmetic encode/decode (`coder.ts` builds the streams) |
| `src/lib/player.svelte.ts` | Playback (index into the trace) |
| `src/components/*` | CostPanel, CostChart, CandidatesTable, Controls (shared) + per-lens views |
| `engine/` | Python engine service (FastAPI + torch): real-model logit lens / J-lens over HF models |

A new domain = implement one `MdlProblem` adapter; the engine, player, and most
of the UI come for free. Lenses sharing this spine today: grammar, morphology
(merge), a SUBDUE graph lens — and, with their own bespoke loops, the Morfessor
split lens and a mini-GPT training trace.

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

## Graph lens (SUBDUE)

Feed it a **directed, labeled graph** as an edge list (`a:Person b:Movie WATCHED`).
This is the grammar lens lifted from a sequence to a graph: instead of naming a
recurring *digram*, it discovers a recurring connected *subgraph* — a triangle, a
ring, a motif — names it, and collapses every non-overlapping (vertex-disjoint)
instance into a single composite node. The folded pattern lives once in the
**substructure dictionary** (that is `L(M)`); the graph that remains is `L(D|M)`.
Adding a substructure grows the dictionary but shrinks the graph — the same MDL
trough, now over a structurally richer object. This is SUBDUE.

A composite node carries a substructure symbol, so a later substructure can
contain an earlier one: the dictionary composes hierarchically, exactly as
grammar builds `"the"` from `(t,h)+e`. The candidates table makes the lesson
sharper than in the string lens — a **large, rarer** substructure can out-compress
a **small, frequent** one, so MDL's pick and the most-frequent pick diverge more
often. Same engine, player, cost panel, candidate table, and evolution chart as
the other lenses; only the data shape (nodes/edges) and the move (collapse a
subgraph) differ.

Notably, this lens needs **no custom engine** — unlike Morfessor and the Mini-GPT
lens. It implements the same `MdlProblem` adapter the grammar lens does and rides
the generic `trace()` runner unchanged; the SUBDUE beam (canonical labeling +
connected-subgraph enumeration + vertex-disjoint instance selection) all happens
inside `candidates()`. That is the strongest evidence yet for the README's thesis:
*a new domain = one adapter.*

Matching is on **labels *and* structure**, not shape alone. Two samples make this
concrete: a **dependency-grammar** graph where the noun phrase (`N←det D`,
`N←amod A`) emerges as a substructure and the clause (`V` with subject- and
object-NPs) then composes on top of it — linguistic constituency falling out of
compression — and a **kinship** graph where a two-son family and a two-daughter
family are the *same shape* but stay *distinct* substructures because sex (M/F) is
part of the match. The all-`o` triangle samples, by contrast, are the graph
equivalent of compressing `aaaaaa`: pure structure, labels carrying nothing.

It is deliberately **toy-scale** (≤ ~40 nodes): subgraph isomorphism is
exponential, so pattern size and the candidate beam are bounded, and an induced
subgraph (a chunk you literally cut out) defines an instance. Node positions come
from a seeded, deterministic force layout computed once over the base graph, with
composites placed at the centroid of their constituents — so nothing jumps as you
scrub.

## Coder lens (arithmetic coding)

Every other lens ends by reducing a model to a per-symbol cost of `−log₂ p` bits.
The coder lens is the **operational back-end** that turns such a probability stream
into an actual interval-narrowing encode/decode round-trip — it makes the
fractional bits *visible* as a literal binary codeword you can decode back.

The contract is deliberately model-agnostic. A `CodeStream` is just an ordered list
of *(distribution at this step, which symbol was emitted)*. The arithmetic core
never asks where the distribution came from:

- **Static sources** — `uniform` (`p = 1/V`), `empirical` (Shannon, `p = count/T`),
  and `grammar` — repeat the **same** distribution every step.
- **A context source** — the Mini-GPT lens — varies the distribution **every step**,
  because position *i*'s softmaxed logits *are* `P(xᵢ₊₁ | x≤ᵢ)`.

Same `encode()`/`decode()` handles both. That single fact is the punchline of the
whole app: **prediction *is* compression.** See "The LLM connection" below.

### How a symbol narrows the interval (encode)

Encoding keeps a live interval `[lo, hi)` inside `[0, 1)`. Each symbol carves out
the sub-slice whose width is its probability `p`, and that slice becomes the new
interval. The *offset* of the slice is the **cumulative mass before** the symbol:

```
width   = hi − lo
newLo   = lo + width · cumLo      // cumLo = Σ p of symbols before this one
newHi   = lo + width · cumHi      // cumHi = cumLo + p
```

After all symbols, the interval width equals `Π p = 2^(−Σ −log₂ p)` — the ideal
bit cost — and **any** number inside it names the message. We emit the
fewest-bit binary fraction in the interval (`shortestCodeword`).

**Worked encode — `"CAB"` over a uniform alphabet `{A,B,C,D}` (`p = ¼` each).**
The four bands of `[0,1)` are `A=[0,.25)`, `B=[.25,.5)`, `C=[.5,.75)`, `D=[.75,1)`:

| step | symbol | interval before | width | symbol's band `[cumLo,cumHi)` | interval after |
|------|--------|-----------------|-------|-------------------------------|----------------|
| 1 | C | `[0, 1)` | 1 | `[0.5, 0.75)` | `[0.5, 0.75)` |
| 2 | A | `[0.5, 0.75)` | 0.25 | `[0, 0.25)` | `[0.5, 0.5625)` |
| 3 | B | `[0.5, 0.5625)` | 0.0625 | `[0.25, 0.5)` | `[0.515625, 0.53125)` |

Final width `= 1/64`, so the ideal cost is `−log₂(1/64) = 6` bits (three symbols ×
`log₂4` = 2 bits each, as expected for a uniform quaternary source). The shortest
dyadic fraction inside `[0.515625, 0.53125)` is `33/64 = 0.515625`, i.e. the
**codeword `0.100001₂`** — six bits.

### How one number replays every offset (decode)

This is the part that feels like magic: the decoder is handed **one number** (the
codeword) plus the symbol count, and recovers every symbol *and* every offset. The
trick is that it never changes the number — it changes its *view* of the number.
At each step it renormalizes the codeword into the current window:

```
x = (value − lo) / width        // where the codeword sits within [lo, hi), as a point in [0,1)
```

Then it walks the cumulative bands of *this step's* distribution and emits whichever
band contains `x` — that lookup *is* the offset. It then narrows `[lo, hi)` to that
band (identical arithmetic to encode) and repeats. The window zooms in on the fixed
point until each symbol is forced.

**Worked decode — codeword `0.100001₂ = 0.515625`, told the message has 3 symbols,**
same `{A,B,C,D}` bands:

| step | interval before | width | `x = (value−lo)/width` | lands in band | emit | interval after |
|------|-----------------|-------|------------------------|---------------|------|----------------|
| 1 | `[0, 1)` | 1 | `0.515625` | C `[0.5, 0.75)` | **C** | `[0.5, 0.75)` |
| 2 | `[0.5, 0.75)` | 0.25 | `0.015625 / 0.25 = 0.0625` | A `[0, 0.25)` | **A** | `[0.5, 0.5625)` |
| 3 | `[0.5, 0.5625)` | 0.0625 | `0.015625 / 0.0625 = 0.25` | B `[0.25, 0.5)` | **B** | `[0.515625, 0.53125)` |

Out comes `"CAB"` — the round-trip closes. Notice the single number `0.515625` was
re-projected to `0.515625`, then `0.0625`, then `0.25`: the *same* point, seen
through three successively magnified windows. That re-projection is the answer to
"where do the offsets come from."

### Unequal bands, fractional bits

When `p` is not `1/V` the bands have **unequal widths** — and that is the only thing
that changes for a skewed or context source. Encode `"ab"` with `p(a)=0.6, p(b)=0.4`
(bands `a=[0,0.6)`, `b=[0.6,1)`):

| step | symbol | interval before | width | band `[cumLo,cumHi)` | interval after |
|------|--------|-----------------|-------|----------------------|----------------|
| 1 | a | `[0, 1)` | 1 | `[0, 0.6)` | `[0, 0.6)` |
| 2 | b | `[0, 0.6)` | 0.6 | `[0.6, 1)` | `[0.36, 0.6)` |

Ideal cost `= −log₂0.6 − log₂0.4 ≈ 0.74 + 1.32 = 2.06` bits — a **fractional** total
no per-symbol integer code (≥ 1 bit per symbol) could reach. The shortest codeword
in `[0.36, 0.6)` is `0.5 = 0.10₂`, two bits.

> **A subtlety worth knowing:** the codeword can come out as short as — occasionally
> a hair under — `⌈Σ −log₂ p⌉`, because `decode()` is *told* the symbol count (it
> loops exactly `stream.length` times); the code is not self-delimiting. Charge for
> the length too and you are back above the Shannon bound. Over a long stream the gap
> vanishes and the codeword length → `Σ −log₂ p`. (Float64 intervals stay exact only
> to ~52 bits, so the lens caps the coded prefix — `MAX_CHARS` / `MAX_BITS` in
> `coder.ts` — to keep the round-trip exact and legible.)

### The LLM connection

This is why the encoder/decoder "also fits" the Mini-GPT lens — it was never a
coincidence. An LLM is a probabilistic model: at each position it emits
`P(next token | context)`. Feed that stream to the coder and the band widths simply
change every step instead of staying fixed. Nothing else differs.

And the total it costs is exactly the master equation's data term:

```
codeword length  ≈  Σᵢ −log₂ P(xᵢ | x<ᵢ)  =  L(D | M)  =  the model's cross-entropy loss (in bits)
```

All three are the *same number*. So an LLM does not "have a codeword" on its own — a
codeword is for a **(model + one specific sequence)** pair: the LLM is the shared
codebook both encoder and decoder hold, and arithmetic coding turns it plus a
sequence into literal bits. Training the LLM to lower its loss is, exactly, training
it to compress. (`llmStream` codes token 0 under a uniform `log₂V`-bit prior, since
the model has no left context to predict the first token, then rides the model's own
distribution for every token after.)

## Transformer lens — fact surgery, logit lens, J-lens

The mini-GPT lens now treats the model as what INSIGHTS.md says it is: a
whitened associative memory you can **write** to, **read** from, and **account
for** in bits.

**Implant facts (WRITE).** The `Implant facts` panel writes a fact into the FFN
in closed form — no training step. The *key* is the FFN's actual input (ln2 of
the residual) at the prompt's last position; the *value* is the target token's
unembed column, centered over the vocabulary; the write is one grafted hidden
unit — `fc1` column = key, `fc2` row = value — with a ReLU threshold gate (θ)
so the memory only fires where the incoming vector aligns with its key. The
`strength` knob is calibrated through the real final LayerNorm to mean "boost
the target's logit by *s*" on every dataset. Implants re-apply to whichever
training step the transport is scrubbed to, so the same surgery can be watched
against untrained, half-trained, and converged weights — early keys are
undifferentiated, and the damage shows.

The interference table prices everything in the house currency: each training
sequence's **description length** `Σ −log₂ p` before → after the implant. Δ<0
on the fact's row is recall working; Δ>0 on an unrelated row is interference —
the key firing where it shouldn't. A `whiten` toggle subtracts the mean
FFN-input from the key before storing (the associative-memory capacity
condition) — and the honest finding is that it often changes little *because
ln2 is already doing it*: the capacity condition is built into the
architecture. Implanting a **novel** prompt is nearly free; **overwriting** a
trained fact is a visible tug-of-war between the grafted memory and the
trained prior.

**Logit lens (READ).** The one-block model has a three-rung residual ladder at
the prediction position — `token+position → +attention → +feed-forward` — and
the `Logit lens` panel decodes each rung through the final LayerNorm + unembed,
reporting `−log₂ p` of the current prediction at every rung. "The prediction
sharpens with depth" becomes the literal MDL statement: *the code for the next
token shortens as the residual stream is refined.* The top rung equals the
model's real output by construction.

**J-lens.** The classic logit lens decodes early residuals *as if the network
were done* — but the remaining layers still transform them. Because the model
is tiny, the panel's `J-lens` mode computes the **exact Jacobian** of the
remaining computation (one reverse-mode autograd sweep per dimension) and
decodes `J·h` instead of `h`: the rung's content transported into the final
basis before reading. Each rung also shows a **visibility** fraction — how much
of the residual's norm lies in the row space of `∂logits/∂h` (centered, since
softmax ignores uniform shifts). The remainder is blind directions: components
that carry exactly zero bits about the next token, however large they are.

## Logit·real lens (GPT-2 / Qwen)

The mini-GPT lens's ladder, climbed by a **real model**. A small Python engine
service (`engine/` — FastAPI + torch, adapted from `x-logit-lens`) runs the
classic logit lens over any cached HF causal LM and reports, per layer, the
**code length in bits** (`−log₂ p`) of the model's own next-token prediction —
so "the prediction sharpens with depth" is the same falling curve, on the same
axis, as every other lens. The transport scrubs through **depth** instead of
training time: the same `Player`, pointed at rungs instead of steps.

Start the engine, then open the `Logit·real` tab:

```bash
cd engine
uv sync                                   # once; pulls torch (CUDA on win/linux)
uv run uvicorn main:app --port 5181       # set HF_HOME first to reuse a model cache
```

Panels: the **lens grid** (rows = layers deepest-on-top, columns = positions,
each cell a layer's top-1 guess shaded by confidence — the x-logit-lens TUI,
reborn), the **depth chart** (bits per rung, classic vs J-lens, uniform log₂V
reference), and the **rung readout** (top-k under both decodes at the selected
cell).

The **J-lens** here mirrors the mini-GPT one *semantically*, but where the toy
model affords an exact Jacobian, the real model gets one **forward-mode AD pass
(JVP) per rung**: seed block *l*'s input with tangent = its own last-position
content, read the tangent back off the pre-norm final residual — that is `J·h`
with `J = ∂(remaining blocks)/∂rung` — then decode it through the *real* final
norm + unembed. (Linearizing the final LayerNorm too would strip the radial
component that carries confidence; that subtlety is why the naive "JVP to the
logits" curve comes out flat.) One extra forward per rung, no reverse graphs,
works unchanged for GPT-2 (`ln_f`) and Llama/Qwen (`model.norm`) layouts.
GPT-2's 13 rungs take under a second on a modest GPU; Qwen2.5-0.5B's 25 take ~3 s.

What it shows: GPT-2 finding ` Paris` around layer 9 and locking in; the J-lens
fixing the classic lens's garbage embed rung (≈51 bits naive → ≈11 transported);
Qwen's mid-stack staying near-illegible to the classic lens while the J-lens
reads the answer many layers earlier — the faithfulness gap, in bits.

## Roadmap

See **[REFERENCE/INSIGHTS.md](REFERENCE/INSIGHTS.md)** for the conceptual
through-line (every lens as a way to write, read, whiten, or account for an
associative memory — all in bits) and
**[REFERENCE/IDEAS.md](REFERENCE/IDEAS.md)** for the dataset backlog: the
tinygrad-style two-digit addition task with a train/held-out split (prequential
MDL made visible), grokking via modular addition, induction-head copy tasks,
city–capital facts for the implant panel, log-file grammar samples,
agglutinative morphology, molecule graphs for SUBDUE — and the WebGPU /
transformers.js notes for a future static-hosting path.

- **Speed (only if real vocabularies are loaded)** — the merge lens scores
  candidates by a full `cost(apply(...))` recompute; both code modes admit an exact
  `O(1)` delta (`L = T·log₂T − Σ f·log₂f` localizes a merge to a few frequencies).
  Worth adding behind a dev-time equality assertion against `cost()`; unnecessary
  for toy inputs.
