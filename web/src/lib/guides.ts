/**
 * Field guides to the instruments: what each reading legitimately supports,
 * and the seductive misreadings. The real-model sections are anchored to
 * observations from this repo's own sessions so the caveats stay memorable;
 * the concept/toy sections play the same ✓/✗ game with the theory.
 *
 * Content lives here (not in the component) so every lens can declare its
 * sections and InterpretGuide stays a pure renderer.
 */

export interface GuideItem {
  ok: boolean; // ✓ valid reading vs ✗ false path
  text: string;
}

export interface GuideSection {
  id: string;
  title: string;
  items: GuideItem[];
}

const SECTIONS: GuideSection[] = [
  // ---- shared MDL machinery (grammar / morph / graph) ----------------------
  {
    id: 'mdlcore',
    title: 'the two-part code (cost panel, candidates table)',
    items: [
      {
        ok: true,
        text: 'Two accounts, one budget: L(M) is the dictionary you would have to transmit; L(D|M) is the data re-spelled using it. A move is good only if the TOTAL falls — every rule is a loan the corpus has to repay.'
      },
      {
        ok: true,
        text: 'The candidates table is ranked by Δbits under the current model — and it flags when MDL’s pick differs from the merely most-frequent candidate. Frequency ≠ compression: a shorter, rarer pattern can save more.'
      },
      {
        ok: false,
        text: 'Rising L(M) is not waste and falling L(D|M) is not the goal — read only the total. The evolution chart’s trough is where the next loan stops paying under this code; that complexity charge is how MDL resists overfitting.'
      },
      {
        ok: false,
        text: 'Convergence is not “the truth about this data” — it is greedy search’s local stop under THIS code. One step deep, no undo: a merge that blocks a better later merge stays blocked.'
      }
    ]
  },
  {
    id: 'codes',
    title: 'code toggles (log₂V vs −log₂p, overhead)',
    items: [
      {
        ok: true,
        text: 'log₂V prices every symbol equally; −log₂p lets frequent symbols get cheap. Switching codes changes which moves are worth it — same data, different decisions, and both are honest codes.'
      },
      {
        ok: false,
        text: 'Overhead off means pretending the decoder already knows the code table — fine for intuition, but the moves it licenses could not actually be transmitted. If a merge only pays with overhead off, it doesn’t really pay.'
      }
    ]
  },
  {
    id: 'morphpair',
    title: 'merge vs split (the two morphology lenses)',
    items: [
      {
        ok: true,
        text: 'Same objective, opposite search: merge grows morphs bottom-up from characters (BPE-style); split starts from whole words and recursively cuts (Morfessor-style). They share sample word lists on purpose — compare them on identical data.'
      },
      {
        ok: false,
        text: 'When the two lenses segment a word differently, neither is wrong — they landed in different local minima of the same cost. The disagreement is the exhibit: greedy search direction is part of the model.'
      },
      {
        ok: true,
        text: 'Frequency drives everything: a merge inside “walking 8” repays its lexicon entry 8× per occurrence. Drop the counts to 1 and watch affixes stop being worth naming.'
      }
    ]
  },
  {
    id: 'graphread',
    title: 'reading the graph lens',
    items: [
      {
        ok: true,
        text: 'Labels are part of the match: a two-son family and a two-daughter family are the same shape but DIFFERENT substructures (the kinship sample exists to show this). Matching is structure + labels, never shape alone.'
      },
      {
        ok: false,
        text: 'The layout is force-directed convenience. Distances, angles, and crossings carry no information — only nodes, labels, and edges do. Two runs of the same graph may look different and mean the same.'
      },
      {
        ok: true,
        text: 'Collapsed nodes can themselves recur: the triangle-chain sample compresses into a chain of triangle-nodes that compresses again. Hierarchy is not assumed anywhere — it falls out of repeated compression.'
      }
    ]
  },
  {
    id: 'coderread',
    title: 'reading the coder',
    items: [
      {
        ok: true,
        text: 'Each symbol narrows the interval by exactly its probability, so bits accrue at −log₂p per symbol and the final codeword length ≈ the ideal total (+ a bit or two of framing). This is “prediction is compression” with no metaphor left.'
      },
      {
        ok: true,
        text: 'Decode replays the SAME distributions the encoder used — which is precisely why MDL charges L(M): a model the decoder doesn’t have isn’t a code, it’s a secret.'
      },
      {
        ok: false,
        text: '“Precision-trimmed” means the demo hit IEEE-754 float resolution, not a flaw in arithmetic coding — production coders renormalize and never run out. Keep demo strings short rather than reading the trim as loss.'
      },
      {
        ok: true,
        text: 'With the Mini·GPT source, scrub the train-step slider: the SAME sequence costs fewer bits under a better-trained model. That falling number is the entire thesis of this app.'
      }
    ]
  },
  {
    id: 'minigpt',
    title: 'reading the Mini·GPT',
    items: [
      {
        ok: true,
        text: 'Scrubbing replays a finished training run against fixed weight snapshots — it is deterministic and free. The probe target’s probability rising means that target’s code is shortening; the loss curve aggregates the priced tokens across the evaluation sample.'
      },
      {
        ok: true,
        text: 'The logit-lens panel here is the honest version of the real one: this readout basis is trained end-to-end with only 2 layers, so mid-depth decodes are meaningful in a way real mid-layer decodes are not (see Logit·real’s ladder caveats).'
      },
      {
        ok: false,
        text: 'Implanting is surgery, not learning: a key→value slot written into the FFN. Check the interference rows before believing a “clean” edit — description length over the WHOLE training set is the damage meter, and edits that look local often aren’t.'
      },
      {
        ok: false,
        text: 'This is a tiny model on a tiny vocabulary: attention here is legible precisely because nothing else is going on. Do not generalize head behavior to real models — that intuition is what Reason·trace’s attention caveats exist to unlearn.'
      }
    ]
  },
  {
    id: 'attnlab',
    title: 'reading the attention lab',
    items: [
      {
        ok: true,
        text: 'This is the mechanism itself, one clickable cell at a time: Q = X·Wq, scores = QKᵀ/√d, softmax rows sum to 1, output = weights·V then ·Wo. If a cell surprises you, click it — the inspector shows the exact dot product.'
      },
      {
        ok: true,
        text: 'The identity preset (Q = K = V = X) is the cleanest way to see why √d scaling exists: raw dot products grow with dimension, and unscaled softmax saturates into one-hot rows.'
      },
      {
        ok: false,
        text: 'Nothing here is learned — random weights produce vivid-looking patterns, and those patterns mean nothing. This lab teaches the computation; what attention patterns may or may not MEAN is the real-model lenses’ topic.'
      }
    ]
  },
  {
    id: 'trainread',
    title: 'reading the training run',
    items: [
      {
        ok: true,
        text: 'Accuracy is held-out exact-match over all three answer digits: 2,000 sums the model never saw. What rises here is generalization (the carry included), not recall.'
      },
      {
        ok: true,
        text: 'The three checkpoints are the point — open them in Logit·real on the same prompt and watch the J-lens ladder go from garbage (step0) to forming (mid) to crisp (final). Training time becomes a lens axis.'
      },
      {
        ok: false,
        text: 'Loss and exact-match move on different schedules: accuracy can sit at 0% while loss falls (partial credit is invisible to exact-match), then jump. The jump is thresholding, not a discovered phase transition.'
      },
      {
        ok: false,
        text: 'This model has 16 positions and a 13-character vocabulary. Prompts must look like “17+25=” — anything else becomes ‘?’ tokens, and prompts past 16 positions are rejected before they can wedge the GPU.'
      },
      {
        ok: true,
        text: 'Stopping mid-run keeps every checkpoint saved so far; re-training overwrites them in place, and the engine drops its cached copies so the next lens call sees the new weights.'
      }
    ]
  },
  // ---- real-model sections (Logit·real / Reason·trace) ---------------------
  {
    id: 'bits',
    title: 'per-token code length (yellow shading, strip chart)',
    items: [
      {
        ok: true,
        text: 'Each token is priced −log₂p under the model’s TRUE distribution. Temperature changes which path is walked — never the prices.'
      },
      {
        ok: true,
        text: 'Spikes are forks: several continuations were live (a plan choice, a discourse pivot, formatting freedom). Valleys are execution: context already determined the token.'
      },
      {
        ok: false,
        text: 'A spike is not confusion or an error. “So” costing 1.9b means the sentence had options, not that the model struggled.'
      },
      {
        ok: false,
        text: 'At temp > 0, a bright token may be an unlucky draw, not a preference — “ violence” was emitted at p=0.34% while the model wanted “ protests” at 38%. Check p before reading intent.'
      },
      {
        ok: true,
        text: 'The sum over the trace is its description length. 512 tokens ≈ 126 bits = the model mostly predicted itself; the information lives in the spikes.'
      }
    ]
  },
  {
    id: 'ladder',
    title: 'classic ladder (logit lens)',
    items: [
      {
        ok: true,
        text: 'Shows WHERE in depth the final prediction crystallizes — good for comparing columns, prompts, models, checkpoints.'
      },
      {
        ok: false,
        text: 'A rung’s decode is not “what layer k believes.” The unembedding was only trained against the final layer; mid-rungs can decode junk (‘oret’, ‘oretical’) while the layer is doing useful work.'
      },
      {
        ok: false,
        text: 'Foreign-vocab tokens mid-ladder (二百 for “two hundred”) are nearest-neighbor artifacts of a language-agnostic concept space — not “thinking in Chinese.” Late layers choose surface form: language, register, spelling.'
      },
      {
        ok: false,
        text: 'Bits rising over the last layers is usually calibration/hedging, not forgetting (gpt2’s Paris: 2.0b at layer 9, 3.8b at final).'
      }
    ]
  },
  {
    id: 'jlens',
    title: 'J-lens',
    items: [
      {
        ok: true,
        text: 'The rung’s content transported through the remaining layers: “what does this layer contribute to the final answer.” Trust it mid-network, where the classic decode is illegible.'
      },
      {
        ok: true,
        text: 'Classic-minus-J gap = information already present but not yet rotated into the readout basis.'
      },
      {
        ok: false,
        text: 'The two curves ALWAYS meet at the final rung — J is the identity there by construction. Their meeting is not a finding.'
      },
      {
        ok: false,
        text: 'It is a linearization (one JVP): readings far from the final layer are extrapolations, and the embed rung’s J decode is usually junk. Exactly tied percentages near a fork are knife-edge numerics, not deep structure.'
      }
    ]
  },
  {
    id: 'attn',
    title: 'attention (purple shading, head grid)',
    items: [
      {
        ok: true,
        text: 'A hypothesis about where the computation looked. The default overlay is value-weighted (a·‖v‖), which already discounts stares at near-zero value vectors.'
      },
      {
        ok: false,
        text: 'Attention weight ≠ importance (“attention is not explanation”). To claim a region mattered, ablate it and read Δbits — weights only nominate candidates.'
      },
      {
        ok: false,
        text: 'Mass on <|im_start|> is an attention sink — a parking spot, not interest. Previous-token and local stripes are positional plumbing. Final-layer heads are mostly retired.'
      },
      {
        ok: false,
        text: 'Diffuse ≠ integrating, focused ≠ important. Before naming a head, click it and see whether its value-weighted gaze concentrates or stays smeared (mean-pooling wears a busy expression).'
      }
    ]
  },
  {
    id: 'ablate',
    title: 'Δbits ablation (re-price without a region)',
    items: [
      {
        ok: true,
        text: 'Causal, in bits: the delta is what reading that region actually bought for THIS token. This is the arbiter the attention overlay defers to.'
      },
      {
        ok: false,
        text: 'Δ≈0 does not mean the reasoning was useless — the visible context may re-derive it. A shown-work answer priced 408 at only +0.11b without the entire think block, because “200+40+140+28 =” was still on screen.'
      },
      {
        ok: false,
        text: 'Masking is off-distribution surgery: a huge mask can derail the computation rather than cleanly remove information. Prefer tight regions and compare against masking a same-sized neutral region.'
      }
    ]
  },
  {
    id: 'repro',
    title: 'sampling & reproducibility',
    items: [
      {
        ok: true,
        text: 'Same seed + temperature replays a trace exactly — within one engine session. The URL carries your settings, so a copied link is a reproduction recipe.'
      },
      {
        ok: false,
        text: 'Across engine restarts, near-tie forks can flip (bf16 kernels aren’t bitwise stable) and the trace diverges from there. Drill-ins use exact token ids, so they never fork.'
      },
      {
        ok: true,
        text: 'Ladder differences under ~0.3b between runs are numeric noise. Don’t interpret them.'
      }
    ]
  },
  {
    id: 'small',
    title: 'small-model humility',
    items: [
      {
        ok: false,
        text: 'This is a 0.6B model: it ruminates under greedy decoding, confabulates details (“the one-child policy” as a protest cause), and holds instructions loosely against trained habits (\\boxed{} beat “just the number”).'
      },
      {
        ok: true,
        text: 'One column is an anecdote. Before concluding, run a control: another topic in the same frame, another seed, or base vs chat on matched context — the deltas are the evidence.'
      }
    ]
  },
  // ---- Tic·arena -------------------------------------------------------------
  {
    id: 'arenaread',
    title: 'reading the arena honestly',
    items: [
      {
        ok: true,
        text: 'LLM moves are read as full next-token distributions over the digit tokens — one forward per position, memoized, no sampling. It measures the model’s PREFERENCE, not one roll of it. Decisiveness is the mass it put on any digit at all; the rest went to off-task tokens.'
      },
      {
        ok: true,
        text: 'Trappy vs uniform solver is the tie-breaking exhibit: identical minimax optimality, different opponent model. Watch trappy convert more wins from random — and watch the two solvers draw each other forever.'
      },
      {
        ok: true,
        text: 'Blocks is defense distilled — positions where the unique optimal move parries a two-in-a-row — and it predicts the round-robin’s O-side column better than agreement does. Agreement rewards matching large optimal sets; blocks has no partial credit.'
      },
      {
        ok: false,
        text: 'Score against the perfect solver only measures “doesn’t lose” — wins against perfect play do not exist, so that column is really a draw rate.'
      },
      {
        ok: false,
        text: 'An LLM’s illegal or occupied-cell mass is not stupidity — the board is serialized text on a task it was never trained for. The report card probes TRANSFER, not intelligence. And chat vs raw prompting are different players: same weights, different question. Quote the mode.'
      },
      {
        ok: false,
        text: 'N games at temperature 0 between deterministic players is one game repeated. Raise the temperature (or vary the seed) before reading a win rate as a statistic.'
      }
    ]
  },
  // ---- Tic·tac ---------------------------------------------------------------
  {
    id: 'tictacarch',
    title: 'the arch × signal grid',
    items: [
      {
        ok: true,
        text: 'Solver-soft targets are the SAME objective as sampled games with the sampling noise removed — the noiseless estimator. If the opening distribution stops wobbling across seeds under `solver`, that wobble was estimator variance, never model failure.'
      },
      {
        ok: true,
        text: 'The MLP is the null model: attention adds no capacity on a fully-observed 9-cell state. What the encoder can buy is weight sharing (symmetry learned once, in cell embeddings, instead of nine times) and READABLE pairwise structure. Compare the equivariance curves at matched params before crediting attention with anything.'
      },
      {
        ok: true,
        text: 'The encoder can attend to empty cells; the gpt arm cannot — no token exists for them, so its board state must emerge internally (the Othello-GPT condition). Same game, two different questions: explicit versus emergent representation.'
      },
      {
        ok: false,
        text: 'Distillation KL falling to ~0 means the student matches the teacher — errors included. Dark knowledge transfers blunders as faithfully as skill; judge the student against the SOLVER (agreement), not just the teacher (KL).'
      },
      {
        ok: false,
        text: 'A tempered teacher (T > 1) is not “more informative truth” — it is a design choice trading gradient signal for target entropy. Quote any distillation result together with its T.'
      }
    ]
  },
  {
    id: 'tictaccircuit',
    title: 'reading the circuit panel honestly',
    items: [
      {
        ok: true,
        text: 'The correlation heatmap is a hypothesis generator: a unit that tracks “X threat on line 012” across the probe suite is a CANDIDATE line detector. Confirm by scrubbing training and watching the correlation strengthen as agreement rises — a detector that predates competence is suspicious.'
      },
      {
        ok: true,
        text: 'L1 makes the weight histogram bimodal, so the threshold slider has a natural notch: compare λ = 0.001 against λ = off and watch “units live” become meaningful instead of arbitrary.'
      },
      {
        ok: false,
        text: 'The sparsity threshold is an inspection choice, not a fact about the model — slide it and watch “the circuit” change size before quoting a unit count anywhere.'
      },
      {
        ok: false,
        text: 'Attention on a cell says where information flowed, not why the move was chosen. Only an ablation supports “this head does X”, and this panel doesn’t ablate.'
      },
      {
        ok: false,
        text: 'High agreement with minimax is behavioral, not mechanistic — the net matches the solved game’s outputs on positions like the ones it saw; there is no game tree inside. “Understands tic-tac-toe” is not a reading this lens supports.'
      }
    ]
  },
  {
    id: 'tictacequiv',
    title: 'symmetry and the two corpora',
    items: [
      {
        ok: true,
        text: 'Nothing in the loss asks for D₄-equivariance — the falling red curve is emergent structure. It falls further on the optimal corpus, where symmetric positions genuinely have symmetric answers.'
      },
      {
        ok: true,
        text: 'The meter compares legal-renormalized policies, so it measures strategy symmetry, not legality bookkeeping. And the probe suite includes positions no optimal game ever visits — agreement is generalization, not memorization.'
      },
      {
        ok: false,
        text: 'Emergent equivariance is approximate: the 8 mini-boards disagree in the tail even late in training. A meter of 0.05 is “roughly symmetric”, never “the network represents the group”.'
      },
      {
        ok: false,
        text: 'The random-legal corpus model is not a worse strategist — it answers a different question. Watch illegal mass fall while agreement stays flat: it learned the rules’ support, not the game’s values. Comparing the corpora is the point of the toggle.'
      },
      {
        ok: false,
        text: 'Uniform mass over all 9 openings is not confusion — all nine are minimax-optimal (perfect play draws from everywhere), so the training target ITSELF is uniform there. Expect ~3.2 bits of irreducible loss at ply 0.'
      }
    ]
  },
  // ---- Hopfield·retrieve ----------------------------------------------------
  {
    id: 'hopfieldread',
    title: 'reading Hopfield·retrieve',
    items: [
      {
        ok: true,
        text: 'One modern update with the stored patterns as both keys and values is EXACTLY one row of softmax attention at β = 1/√d — same arithmetic as Attn·lab, no analogy. A head retrieves in one step because one step is already ε-close: that cliff in the energy chart is the paper’s one-step-convergence theorem, drawn.'
      },
      {
        ok: true,
        text: 'Metastable ≠ failure. When several patterns are close, the fixed point is their weighted average — that’s pooling / prototype formation, and it’s what many real attention heads do on purpose. Read the entropy badge (eff. patterns = 2^H), not just the picture.'
      },
      {
        ok: true,
        text: 'Low β doesn’t break the network, it changes the question: from “which pattern is this?” toward “what do my patterns have in common?”. The regime-vs-β chart is one memory answering three different questions.'
      },
      {
        ok: false,
        text: 'The energy is guaranteed to go downhill (CCCP), but downhill to the NEAREST fixed point — not necessarily the pattern you meant. A converged trace showing the wrong glyph is convergence working correctly on a bad query.'
      },
      {
        ok: false,
        text: 'Don’t read the classical net as merely broken: sign(Wξ) with Hebbian weights is the historical baseline whose failure modes — spurious mixtures, ~0.14·d capacity — are precisely what the lse energy fixes. The contrast is the content.'
      }
    ]
  },
  {
    id: 'hopfieldheads',
    title: 'reading Hopfield·heads',
    items: [
      {
        ok: true,
        text: 'γ rescales the softmax temperature without touching the weights: softmax(γ·z) = wᵞ/Σwᵞ from the row the model already computed. γ = 1 IS the model — every other γ is a counterfactual asking how close this head sits to a phase boundary.'
      },
      {
        ok: true,
        text: 'A sharp retrieval head at γ = 1 that stays sharp down to γ = 0.25 is robustly retrieving (well-separated patterns); one that melts immediately was barely deciding. Distance-to-boundary is the reading, not just the color at γ = 1.'
      },
      {
        ok: false,
        text: 'A “global” head is not a broken head — uniform averaging over context is a feature many heads are trained into (and the paper found most heads in middle layers metastable on purpose). Regime is a description, not a grade.'
      },
      {
        ok: false,
        text: 'This reads WHERE a head looks, not whether it mattered. Attention is not explanation: to claim the retrieved position mattered, ablate it (Reason·trace’s Δbits) — regimes only nominate candidates.'
      },
      {
        ok: true,
        text: 'The stored patterns here are the sequence’s own positions (keys), and the retrieval space is learned (W_Q/W_K ≠ identity) — this is the toy lens’s update run inside a projected space where keys ≠ values.'
      }
    ]
  },
  {
    id: 'hopfieldcap',
    title: 'reading the capacity panel',
    items: [
      {
        ok: true,
        text: 'Exponential capacity (the paper’s c·2^(d/2)-flavored results) is for RANDOM, well-separated patterns on the sphere. The random-±1 curve is the theorem’s home turf: watch modern stay near 1.0 while classical collapses past ~0.14·d.'
      },
      {
        ok: false,
        text: 'Correlated patterns void the warranty: glyphs share strokes, so glyph capacity dies far below any exponential promise. The separation Δ (patterns panel) is the quantity the theorems actually charge for, not raw N.'
      },
      {
        ok: false,
        text: 'A success rate of 1.0 at N = 128 is not “infinite memory” — it’s this noise level, this β, this Δ. Raise the noise slider and re-run before believing anything.'
      }
    ]
  }
];

export const GUIDE_SECTIONS: Record<string, GuideSection> = Object.fromEntries(
  SECTIONS.map((s) => [s.id, s])
);
