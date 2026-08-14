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
        text: 'Rising L(M) is not waste and falling L(D|M) is not the goal — read only the total. The evolution chart’s trough is where the next loan stops paying, which is exactly where MDL stops overfitting.'
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
        text: 'Scrubbing replays a finished training run against fixed weights snapshots — it is deterministic and free. The probe’s target probability rising IS the description length of the dataset falling.'
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
  }
];

export const GUIDE_SECTIONS: Record<string, GuideSection> = Object.fromEntries(
  SECTIONS.map((s) => [s.id, s])
);
