# Math Foundations area: review and implementation plan

## Purpose

`trace·lab` makes difficult ideas visible, but it currently assumes that the
reader already knows why the quantities on screen are legitimate. The proposed
`Math·foundations` area should supply that missing bridge.

It is not a glossary and not a conventional textbook. It is a short sequence of
derivations, each attached to a tiny manipulable example and linked back to the
exact readouts in the exhibits. Its main promise is:

> Every important number in the app can be followed from ordinary counting to
> the displayed formula, with its units, assumptions, and limitations made
> explicit.

The primary learning path should make this chain feel inevitable:

```text
counting choices
  -> logarithms
  -> information in one outcome (surprisal)
  -> average information (entropy)
  -> codeword length
  -> arithmetic coding
  -> conditional probability and the chain rule
  -> cross-entropy / negative log-likelihood
  -> training as compression
  -> model cost and MDL
```

The remaining chapters branch from that spine into the linear algebra,
calculus, statistics, symmetry, graph theory, and dynamical systems needed by
the neural and combinatorial lenses.

## What the current project actually covers

The review followed the formulas and measurements in `web/src/lib`, the real
model calculations in `engine/lens.py`, the UI explanations in
`web/src/lib/guides.ts`, and the long-form descriptions in `docs/lenses.md`.

### Concept inventory by exhibit

| Exhibit | Explicit mathematics | Prerequisites currently left implicit |
|---|---|---|
| Grammar | `log2(V)`, `-log2(p)`, two-part MDL, empirical entropy, greedy `delta L` | logarithms, probability distributions, expected length, valid codes, local vs global optimization |
| Morph·merge / Morph·split | weighted counts, entropy-coded tokens, exhaustive segmentation, coordinate descent | type/token distinction, combinatorics (`2^(n-1)` cuts), conditional dependence through shared counts, convergence to a local minimum |
| Graph·SUBDUE | graph description length, canonical labeling, induced connected subgraphs, greedy disjoint instances | graph isomorphism, permutations, combinatorial explosion, maximal vs maximum sets, code-design assumptions |
| Coder | cumulative distributions, affine interval maps, products of probabilities, dyadic fractions | binary expansions, prefix/self-delimiting codes, why widths multiply, why logs turn products into sums, finite precision |
| Mini·GPT / Train·real | logits, softmax, cross-entropy, backpropagation, Adam, residuals, LayerNorm, Jacobians, projections | vectors and matrices, derivatives and chain rule, expectation and sampling, mean loss vs total code length, nats vs bits |
| Attn·lab | dot products, matrix multiplication, `QK^T/sqrt(d)`, row softmax, convex combinations | variance of a sum, exponentials, normalization, causal masks, basis/projection language |
| Logit·real / Reason·trace | token surprisal, conditional probabilities, logit lens, JVP/Jacobian linearization, likelihood-ratio ablation | autoregressive factorization, log probabilities, calibration, first-order approximation, causal interventions |
| Hopfield·retrieve / Hopfield·heads | energy functions, softmax retrieval, inverse temperature, entropy, `2^H`, cosine similarity, fixed points | norms, log-sum-exp, iterative maps, Lyapunov/energy descent, associative memory, phase/regime language |
| Tic·tac / Tic·arena | minimax, distributions over actions, soft targets, KL, total variation, D4 equivariance, L1, Pearson correlation, PCA | recursion and dynamic programming, game trees vs DAGs, group actions, regularization, correlation vs causation, eigenvectors |

### Cross-cutting concepts that recur most often

These deserve first-class lessons rather than repeated tooltips:

1. quantities and units: probability, log probability, nats, bits, bits/token;
2. logs and exponentials: base change, products becoming sums, log-odds;
3. discrete distributions: normalization, marginal/conditional/joint probability;
4. expectation and averages: event surprisal vs entropy vs dataset loss;
5. coding: fixed-length, prefix-free, ideal fractional lengths, arithmetic codes;
6. entropy, cross-entropy, KL divergence, perplexity, effective support size;
7. the probability chain rule and autoregressive sequence likelihood;
8. two-part codes and Minimum Description Length;
9. vectors, matrices, dot products, norms, cosine similarity, projections;
10. softmax, temperature, logits, and numerical stability;
11. derivatives, gradients, chain rule, backpropagation, Jacobians and JVPs;
12. optimization: greedy descent, coordinate descent, SGD/Adam, local minima;
13. regularization and sparsity, especially L1 and its subgradient;
14. statistics: samples, estimators, train/test splits, variance and controls;
15. graphs, permutations, isomorphism, recursion, and combinatorial complexity;
16. symmetry: groups, actions, orbits, invariance, equivariance, pullbacks;
17. dynamical systems: iterative updates, energy, fixed points, convergence;
18. causal evidence: attention weights, ablation, confounding and controls.

## The core derivation spine

The first part of the area should derive the house currency rather than merely
define it. Each displayed equality below should become one short lesson.

### 1. Why a fair choice costs `log2(N)` bits

A `b`-bit binary string distinguishes `2^b` possibilities. To distinguish `N`
equally likely possibilities, solve

```text
2^b = N  =>  b = log2(N).
```

When `N` is not a power of two this is an ideal average/fractional length, not
necessarily the length of one standalone binary word. This is the correct
starting point for the app's uniform `log2(V)` code.

### 2. Why one event carries `-log2(p)` bits

There are two complementary derivations worth showing.

**Counting argument.** An event with probability `p` occupies roughly one out
of `1/p` equally likely cases, so identifying it costs

```text
log2(1/p) = -log2(p).
```

**Additivity argument.** Independent events multiply probabilities. If
information should add when events combine, then `I(pq) = I(p) + I(q)`. The
continuous solutions have the form `I(p) = -k log(p)`. Choosing base 2 and
`I(1/2) = 1` sets `k = 1`, giving bits.

This quantity is **self-information** or **surprisal**. It belongs to a realized
outcome, not to a distribution as a whole:

```text
p = 1       -> 0 bits
p = 1/2     -> 1 bit
p = 1/8     -> 3 bits
p = 0.01    -> about 6.64 bits
```

### 3. Why entropy is measured in bits

Entropy is the expected surprisal of one draw `X ~ p`:

```text
H(p) = E_p[-log2 p(X)] = -sum_x p(x) log2 p(x).
```

It is measured in bits only because the logarithm is base 2. Natural logs give
nats; base-10 logs give hartleys. The conversion is

```text
bits = nats / ln(2),       nats = bits * ln(2).
```

The area must distinguish these commonly conflated values:

- surprisal: one observed outcome, `-log2 p(x)` bits;
- entropy: the average surprisal when reality follows `p`, bits/outcome;
- total information: a sum over observations, bits;
- entropy rate: long-run average for a dependent sequence, bits/token.

### 4. Why entropy is related to codeword length

For a binary prefix code with integer lengths `l_x`, the Kraft inequality says

```text
sum_x 2^(-l_x) <= 1.
```

This is the mathematical bridge between a distribution and decodable binary
strings. Ideal lengths `l_x = -log2 p(x)` satisfy Kraft with equality because
`sum_x p(x) = 1`. Integer prefix codes usually use rounded lengths; Shannon and
Huffman codes achieve an average below `H(p) + 1` bit per symbol.

Arithmetic coding avoids rounding each symbol separately. A sequence
`x_1,...,x_n` receives an interval whose width is

```text
width = product_i p(x_i | x_<i).
```

Therefore its ideal binary precision is

```text
-log2(width)
  = -log2 product_i p(x_i | x_<i)
  = sum_i -log2 p(x_i | x_<i).
```

This derivation should be paired directly with the existing Coder interval
view. It answers both “why fractional bits?” and “how can probabilities become
literal bits?”

### 5. Conditional probability and sequence likelihood

The probability chain rule is

```text
p(x_1,...,x_n)
  = p(x_1) p(x_2|x_1) ... p(x_n|x_<n).
```

Taking `-log2` turns the product into the sum shown throughout the app:

```text
-log2 p(sequence) = sum_i -log2 p(x_i | x_<i).
```

An autoregressive language model supplies those conditional distributions. It
does not itself emit a compressed file; paired with an entropy coder and the
same model at the decoder, it defines one.

### 6. Cross-entropy as expected code length

Let `q` be the data-generating distribution and `p` the model used as a code.
The expected number of bits paid under the model is

```text
H(q,p) = E_q[-log2 p(X)] = -sum_x q(x) log2 p(x).
```

That is cross-entropy. For a one-hot observed target `y`, the sum collapses to

```text
loss = -log2 p(y).
```

So token cross-entropy, token surprisal, and ideal token code length are the
same numerical quantity when they use the same log base. The implementation's
training loss often uses natural logs, so it is in nats until divided by
`ln(2)`.

Dataset **mean** cross-entropy is bits/token. Dataset description length is the
**sum**, or equivalently mean times the number of priced tokens. This distinction
must be visible wherever the app says “loss = description length.”

### 7. Why minimizing cross-entropy works

Cross-entropy decomposes as

```text
H(q,p) = H(q) + KL(q || p),

KL(q || p) = sum_x q(x) log2(q(x)/p(x)) >= 0.
```

For fixed data distribution `q`, `H(q)` cannot be changed. Training can only
reduce the extra cost `KL(q || p)`. The non-negativity of KL can be derived with
the log inequality or Jensen's inequality in an optional proof drawer.

This also explains distillation:

```text
cross-entropy(teacher, student)
  = entropy(teacher) + KL(teacher || student).
```

The student can drive the KL term to zero while faithfully copying the
teacher's mistakes.

### 8. From maximum likelihood to compression and MDL

For fixed model `M`, minimizing negative log-likelihood is minimizing the data
code:

```text
argmin_M [-log2 p(D|M)] = argmax_M p(D|M).
```

If model complexity is free, a sufficiently flexible model can memorize the
sample. MDL charges for both parts:

```text
L(M,D) = L(M) + L(D|M).
```

This is the right place to explain the relationship to regularization and a
Bayesian MAP objective: a code length `L(M)` corresponds to a prior proportional
to `2^-L(M)` when it is a valid code. These are related viewpoints, not an
assertion that every hand-built penalty is automatically a valid model code.

### 9. Derived quantities used throughout the app

The foundations area should derive these from the same definitions:

```text
perplexity              = 2^(average bits/token)
effective possibilities = 2^H
log-odds(a vs b)        = log(p(a)/p(b))
delta bits after ablation
  = -log2 p_masked(y) + log2 p_base(y)
  = log2[p_base(y) / p_masked(y)]
```

Thus `delta bits` is a log likelihood ratio for one chosen token under two
interventions. It is causal evidence about that intervention on that example,
not a general measure of a region's meaning.

## Proposed information architecture

Add `Math·foundations` as a top-level destination beside the exhibit index, not
as another `LensMeta` entry. It is a reference/course area and should not be
misclassified as a concept exhibit with a playback trace.

Suggested route:

```text
#/math                    overview and dependency map
#/math/<chapter>/<lesson> one stable, linkable lesson
```

The existing router currently treats only one path segment as an exhibit id.
The implementation can either preserve the rest of the path as a string or use
query state (`#/math?lesson=surprisal`) for the first version. Nested paths are
cleaner and should be preferred if the router change remains small.

### Learning tracks

#### Track A — The house currency (required, lessons 1–9)

1. Choices and binary numbers
2. Logarithms without mystery
3. Probability and conditional probability
4. Surprisal: information in one event
5. Entropy: average surprisal
6. Codes, Kraft's inequality, and arithmetic coding
7. Cross-entropy, KL, and perplexity
8. Sequence probability and language models
9. Two-part codes and MDL

Finishing this track should make every “bits” readout in the app interpretable.

#### Track B — Neural computation

1. Scalars, vectors, matrices, shapes
2. Dot products, norms, cosine, and projections
3. Linear maps, embeddings, and logits
4. Exponentials, softmax, and temperature
5. Attention from `X` through `Q/K/V` to output
6. Residual streams and LayerNorm
7. Derivatives, gradients, and the chain rule
8. Backpropagation and gradient descent
9. Adam, sampling noise, and train/eval loss
10. Jacobians, JVPs, and first-order linearization
11. L1 regularization, sparsity, and subgradients

#### Track C — Structure and search

1. Greedy search, coordinate descent, and local minima
2. Combinatorics of segmentation
3. Graphs, induced subgraphs, and isomorphism
4. Canonicalization by permutations
5. Maximal versus maximum non-overlapping matches
6. Computational complexity and bounded toy searches

#### Track D — Memory and dynamics

1. Iterative maps and fixed points
2. Energy/Lyapunov functions and convergence
3. Classical Hopfield energy and Hebbian outer products
4. Modern Hopfield updates and log-sum-exp
5. Attention as one retrieval update
6. Inverse temperature, entropy, and retrieval regimes
7. Capacity, separation, noise, and empirical success rates

#### Track E — Games, symmetry, and evidence

1. Recursion, dynamic programming, and minimax
2. Policies, values, optimal sets, and soft targets
3. Groups and the eight elements of `D4`
4. Group actions, orbits, invariance, and equivariance
5. Total variation distance and policy pullbacks
6. Pearson correlation and why it is not mechanism
7. PCA and eigenvector projections
8. Ablations, controls, and causal claims
9. Sampling, estimator variance, held-out evaluation, and generalization

### Two ways to enter

The overview should support both:

- **Learn in order** — start at binary choices and follow prerequisites;
- **Explain this number** — choose a UI readout such as `3.2 bits`, entropy,
  loss, KL, `delta bits`, attention weight, or equivariance error and jump to
  its derivation.

This second entry point fits the user's actual experience: recognizing a symbol
in an exhibit and wanting to know what makes it legal.

## Lesson design

Every lesson should use the same compact structure:

1. **Question** — the plain-language confusion it resolves;
2. **Objects and units** — define every symbol before using it;
3. **Small example** — no more than four outcomes or three tokens;
4. **Derivation** — one equality per line, with a reason beside each step;
5. **Manipulable check** — a slider/table where the arithmetic recomputes;
6. **What it means** — one paragraph;
7. **What it does not mean** — one common overclaim;
8. **Seen in trace·lab** — deep links to the relevant exhibits;
9. **Check yourself** — one prediction before revealing the answer;
10. **Further proof** — optional drawer for the rigorous argument.

The prose should never introduce two new abstractions in one sentence. Formula
symbols should be clickable or hoverable to reveal their definition and units.

### Flagship interactive lessons

Build these first because they answer the motivating questions directly.

#### A. Probability to bits table

A slider for `p` shows `p`, `1/p`, `-log2(p)`, odds, and a visual binary tree.
Include exact landmarks `1`, `1/2`, `1/4`, `1/8` before decimals.

#### B. Entropy mixer

Two to four adjustable probabilities. Show each term
`p_i * (-log2 p_i)`, their sum, the maximum `log2(V)`, and `2^H` effective
choices. This distinguishes a surprising event from an uncertain distribution.

#### C. Code tree and Kraft budget

Let the learner assign binary codewords. Highlight prefix collisions and show
the Kraft sum as occupied capacity in a binary tree. Then compare integer
codeword lengths with the fractional ideals `-log2(p)`.

#### D. Arithmetic interval microscope

Reuse the existing arithmetic-coder core and interval view, but annotate every
step with the invariant `new width = old width * p`. The final panel should
explicitly take `-log2` of both sides to reveal the sum of token surprisals.

#### E. Cross-entropy workbench

Hold the true/target distribution `q` fixed while the learner adjusts model
`p`. Display `H(q)`, `H(q,p)`, and `KL(q||p)` as stacked lengths. Switch between
a one-hot target and a soft target. This single exercise connects classification,
language modeling, solver-soft training, and distillation.

#### F. Autoregressive code receipt

Use a three-token sequence. Show the model's conditional distribution before
each token, the charged surprisal, the product probability, and the accumulated
sum. A toggle between product-space and log-space should visibly demonstrate
why implementations keep log probabilities.

#### G. Model/data balance scale

A tiny dictionary compressor with one candidate rule. Show separately what the
rule costs and what its uses save. Let the user increase occurrence count until
the rule “repays its loan.” Link to Grammar and both morphology lenses.

#### H. Softmax and temperature

Three logits, with a shared-offset control demonstrating
`softmax(z + c) = softmax(z)`. Show logit differences/log-odds, stable
max-subtraction, and temperature. Explicitly separate the sampling distribution
from the untempered distribution used to price tokens in Reason·trace.

#### I. Dot product to attention

Two-dimensional queries and keys on a plane, followed by the exact dot product,
scaling, row softmax, and weighted value sum. Then link into Attn·lab for the
full matrix version.

#### J. Gradient of cross-entropy

For a three-class softmax derive

```text
d loss / d logit_j = p_j - 1[j=y]
```

and show one gradient step moving the target logit up and alternatives down.
This is the smallest honest answer to “how can training actually work?”

## Exhibit-to-foundation links

Every exhibit should gain a small `math behind this` link, contextualized to
the current panel where possible.

| Current readout | Foundation destination |
|---|---|
| `log2(V)` toggle | fair choices and fixed-length codes |
| `-log2(p)` / token shading | surprisal |
| entropy badge | entropy and effective choices |
| ideal bits vs codeword bits | arithmetic coding and integer framing |
| `L(M) + L(D|M)` | two-part codes and MDL |
| loss curve | cross-entropy, units, mean vs total |
| KL teacher/student | cross-entropy decomposition |
| logits / probability bars | softmax and log-odds |
| attention matrix | dot products, scaling, softmax, convex sums |
| classic/J-lens ladder | projections, Jacobians, linearization |
| visibility fraction | row space and orthogonal projection |
| `delta bits` ablation | log likelihood ratios and intervention caveats |
| Hopfield energy | fixed points and energy descent |
| `2^H` effective patterns | entropy and perplexity/effective support |
| Pearson heatmap | covariance, standardization, correlation caveats |
| PCA scatter | centering, covariance, eigenvectors and projection |
| D4 panels | group actions, orbits, equivariance, TV distance |
| L1 controls | absolute-value penalty and sparsity |
| minimax agreement | recursion, values, policies, optimal sets |

## Mathematical precision audit

The foundations work should be accompanied by a small language and accounting
cleanup. None of these invalidate the exhibits; they distinguish a pedagogical
model from a theorem or production code.

### Statements to revise

1. **“MDL never overfits.”** MDL penalizes complexity and is designed to resist
   overfitting, but a chosen code, finite sample, approximate search, or model
   class can still generalize poorly. Prefer: “MDL makes complexity pay for
   itself in the objective.”

2. **“The model that predicts best is the model that compresses best.”** This is
   exact for the same observations, probability assignment, log base, and shared
   side information. For model selection, model/transmission cost also matters.

3. **“Loss is description length.”** A one-token base-2 NLL is an ideal token
   length. A mean loss is bits/token; multiply by the number of priced tokens for
   a total. Natural-log training losses are nats until converted.

4. **“Whiten.”** The implant code subtracts a mean. That is centering, not full
   whitening (which also decorrelates and rescales by covariance). Rename the
   toggle or explicitly call it “mean-center (whitening approximation).”

5. **J-lens transport.** `J h` is a tangent-based transported readout, not the
   full affine first-order approximation `F(h0) + J(h-h0)` and not an exact
   nonlinear continuation. The real-model guide already calls it a
   linearization; the foundations lesson should make the missing intercept and
   locality concrete.

6. **Attention entropy.** Low entropy means a concentrated row, not importance,
   information flow, or causal effect. Keep the existing ablation caveat and
   connect it to the evidence lesson.

### Coding/accounting assumptions to expose

1. The `-log2(p)` lengths in Shannon mode are ideal fractional lengths. They are
   not generally standalone integer prefix codewords; arithmetic coding realizes
   them asymptotically over a message.

2. The Coder demo is told the symbol count, so its displayed binary fraction is
   not self-delimiting. The message length and model/distributions are side
   information. The current long-form documentation notes this; the UI should
   link to it.

3. The grammar/morph/graph cost functions are transparent teaching codes, not
   complete universal codes. Several framing choices are simplified. In
   particular, fixed `charBits = 8`, graph label spellings, counts, collection
   sizes, and some boundaries are not fully transmitted by every displayed
   total. Label them “the exhibit's code” rather than an absolute number of bits
   inherent in the data.

4. Turning overhead off intentionally produces an incomplete transmission
   account. Keep it as an intuition control, but visually mark the total as
   “idealized / decoder given the table.”

5. An empirical distribution estimated from the same message is not free. The
   current code-table charges are simple approximations; a rigorous treatment
   would specify an integer/universal code for counts and framing.

6. The shared `surprisal(p)` helper returns `0` when `p <= 0`. Mathematically a
   zero-probability observed event has infinite surprisal. Current call sites
   generally pass positive observed frequencies, but the helper should either
   return `Infinity`, throw, or document that it only accepts positive `p`.

### Optimization and evidence caveats to preserve

- Grammar, morphology, and graph searches find local optima under their current
  move sets and code, not a uniquely true structure.
- Graph instance selection is maximal, not maximum; the latter is an NP-hard
  independent-set problem on the overlap graph.
- Correlation and attention are hypothesis generators. Ablation is stronger
  causal evidence but is still an off-distribution intervention.
- Exact-match accuracy, average loss, calibration, and sequence-level code
  length are different measurements and need not move together.
- Sampling temperature changes the generated path. If reported probabilities
  remain untempered, it does not change the price function used on that path.

## Implementation shape

### Content model

Keep lesson content separate from rendering, as `guides.ts` already does for
field guides. A typed registry could look like:

```ts
interface MathLesson {
  id: string;
  chapter: string;
  title: string;
  question: string;
  prerequisites: string[];
  terms: { symbol: string; meaning: string; unit?: string }[];
  sections: LessonBlock[];
  labs: MathLabId[];
  seenIn: { lens: string; panel?: string; note: string }[];
}
```

Suggested files:

```text
web/src/lib/math/types.ts
web/src/lib/math/lessons.ts
web/src/lib/math/derive.ts       pure calculators used by interactives
web/src/components/math/MathApp.svelte
web/src/components/math/ChapterNav.svelte
web/src/components/math/Lesson.svelte
web/src/components/math/Equation.svelte
web/src/components/math/labs/*
```

Pure calculators should be unit tested. Reuse the arithmetic coder, attention
helpers, chart scale, colors, and probability conventions instead of creating
parallel implementations.

### Formula rendering

The current app has no math renderer. For multi-line derivations, matrices, and
accessible semantics, add KaTeX (or an equivalently small static renderer) and
wrap it in one `Equation` component. The wrapper should provide:

- a readable text/ARIA description;
- display and inline modes;
- defined-symbol metadata;
- graceful plain-text fallback;
- no user-provided formula evaluation.

If dependency-free delivery is a priority, phase 1 can use semantic HTML and
monospace equations, but the attention/Jacobian chapters will become hard to
read quickly.

### Navigation and layout

The current top navigation is already crowded. Add one prominent `Math` link
beside the wordmark rather than another group of 30 lesson tabs. Inside the
area use:

- a collapsible chapter rail;
- one scrollable lesson column;
- a sticky “symbols in this lesson” card on wide screens;
- previous/next prerequisite navigation;
- `seen in` links that return to an exhibit;
- URL-addressable lesson and optional lab state.

On a narrow screen, the symbol card collapses beneath the derivation and the
chapter rail becomes a drawer.

### Reuse versus extraction

Initially import pure functions from their existing homes. Extract a shared
primitive only when the foundations lab and exhibit genuinely use the same
contract. Strong reuse candidates are:

- `uniformBits`, corrected `surprisal`, and `entropy`;
- arithmetic `encode`, `decode`, and `shortestCodeword`;
- `softmaxVec` and small tensor operations;
- tic-tac-toe transforms and solver;
- Hopfield entropy/regime helpers.

Do not make educational labs depend on large Svelte exhibit components. Their
state and explanatory sequence are different even when the calculation is the
same.

## Delivery phases

### Phase 0 — precision pass

- revise the overclaims and unit labels listed above;
- decide and document which displayed totals are idealized teaching codes;
- correct the zero-probability surprisal contract;
- standardize labels: `bits`, `bits/token`, `nats/token`, and `total bits`;
- add a shared tooltip for “ideal code length.”

### Phase 1 — answer the motivating questions

Ship Track A and flagship labs A–G. Add `math behind this` links to Grammar,
Morph, Coder, Mini·GPT loss, Logit·real, and Reason·trace. This phase alone
should answer entropy, code length, surprisal, cross-entropy, sequence logprobs,
and MDL end to end.

### Phase 2 — explain neural computation

Ship Track B and labs H–J. Link Attn·lab, Mini·GPT activations/training,
Logit/J-lens, Train·real, L1, PCA, and fact implants.

### Phase 3 — explain the specialized exhibits

Ship Tracks C–E. Add graph/isomorphism, Hopfield/dynamics, minimax, symmetry,
metrics, and causal-evidence lessons.

### Phase 4 — integration polish

- contextual deep links from individual panel headers;
- search by symbol (`H`, `KL`, `beta`, `delta bits`) and plain phrase;
- progress stored locally, never required for navigation;
- “show prerequisite” popovers;
- print-friendly lesson mode;
- accessibility and keyboard pass;
- content tests that ensure every registered lesson and lens link resolves.

## Acceptance criteria

The area is successful when a reader who recognizes formulas but cannot yet
derive them can do all of the following without leaving the app:

1. explain why base-2 information is measured in bits;
2. compute surprisal for `p = 1/2`, `1/4`, and `1/8`;
3. distinguish surprisal, entropy, cross-entropy, KL, and perplexity;
4. derive the arithmetic-coding width/product/sum relationship;
5. explain why a model and decoder must share the same distributions;
6. turn an autoregressive sequence probability into a sum of token costs;
7. state when cross-entropy is bits/token and when it is nats/token;
8. explain why minimizing NLL improves compression on the priced data;
9. explain why model cost is needed for model selection;
10. follow a dot product through softmax attention;
11. understand at a high level how a gradient changes a logit;
12. distinguish an attention/correlation observation from an ablation claim;
13. interpret `2^H`, `delta bits`, KL, TV distance, and D4 equivariance in the
    exhibits where they appear;
14. name the simplifying assumptions behind any displayed “bits” total.

## Recommended first implementation slice

Build one vertical slice before authoring the entire curriculum:

1. add the top-level Math route and chapter shell;
2. implement lessons 1–7 of Track A;
3. build the probability-to-bits, entropy mixer, arithmetic interval, and
   cross-entropy labs;
4. add contextual links from Coder, Reason·trace, and one training loss chart;
5. test the flow with the exact question: “Why is entropy measured in bits, and
   what does that have to do with cross-entropy loss?”

That slice proves the content system, formula rendering, interactive pattern,
deep linking, units, and reuse of existing calculations. It also delivers the
highest-value conceptual bridge before the more specialized mathematics is
written.
