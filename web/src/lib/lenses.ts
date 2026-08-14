/**
 * The lens registry — the single description of what lenses exist, mirroring
 * the MdlProblem-adapter ethos on the UI side: the nav, the router, the index
 * page, and the guide cross-links all read this instead of hand-maintained
 * lists. Adding a lens = one entry here + one component mapping in App.svelte
 * (kept separate so this module stays import-cycle-free: components import
 * this file for metadata).
 *
 * `kind` is a real distinction, not just a nav grouping:
 *  - concept:    closed-universe exhibits of the theory — fixed samples, light
 *                URL state, everything computed in the browser.
 *  - toy:        hand-sized neural models you can poke; closed today, but the
 *                natural home for a principled dataset registry later.
 *  - instrument: open-ended workbenches over real models via the engine
 *                service — engine store, full settings-in-URL (seeds included),
 *                the richest "how to read this" content.
 */

export type LensKind = 'instrument' | 'toy' | 'concept';

export interface LensMeta {
  id: string;
  /** Nav label. House style: middot compounds (Morph·merge, Logit·real). */
  title: string;
  kind: LensKind;
  /** One-sentence card text for the index page. */
  blurb: string;
  /** Needs the engine service (engine/, port 5181). */
  engine?: boolean;
  /** Related lenses — instruments point down at the concept/toy that isolates
   * their mechanism; siblings point at each other. Rendered as "see also". */
  seeAlso?: string[];
}

export const KIND_ORDER: LensKind[] = ['instrument', 'toy', 'concept'];

export const KINDS: Record<LensKind, { label: string; tagline: string }> = {
  instrument: {
    label: 'instruments',
    tagline: 'open-ended workbenches over real models — needs the local engine'
  },
  toy: {
    label: 'toy models',
    tagline: 'hand-sized neural nets, every number visible and pokeable'
  },
  concept: {
    label: 'concepts',
    tagline: 'closed exhibits of the theory — one bits-saving move at a time'
  }
};

export const LENSES: LensMeta[] = [
  {
    id: 'logit',
    title: 'Logit·real',
    kind: 'instrument',
    engine: true,
    blurb:
      'The classic logit lens + J-lens over real GPT-2 / Qwen, in bits — watch the prediction crystallize with depth.',
    seeAlso: ['llm', 'train', 'reason']
  },
  {
    id: 'reason',
    title: 'Reason·trace',
    kind: 'instrument',
    engine: true,
    blurb:
      'Stream a Qwen3 thinking trace live: per-token code length, depth ladders, attention, and Δbits ablation of the think block.',
    seeAlso: ['logit', 'attn', 'coder']
  },
  {
    id: 'train',
    title: 'Train·real',
    kind: 'instrument',
    engine: true,
    blurb:
      'Train the tiny addition GPT-2 on the engine and watch loss and held-out accuracy live — the checkpoints land in Logit·real’s model picker.',
    seeAlso: ['logit', 'llm']
  },
  {
    id: 'llm',
    title: 'Mini·GPT',
    kind: 'toy',
    blurb:
      'A 2-layer GPT trained in your browser: scrub training, read every activation, implant facts straight into the FFN.',
    seeAlso: ['attn', 'logit', 'coder']
  },
  {
    id: 'attn',
    title: 'Attn·lab',
    kind: 'toy',
    blurb:
      'Scaled dot-product attention on editable Q/K/V matrices — click any cell and see exactly where the number came from.',
    seeAlso: ['llm', 'reason']
  },
  {
    id: 'grammar',
    title: 'Grammar',
    kind: 'concept',
    blurb:
      'RePair over any string: repeatedly name the digram that saves the most bits and watch a rule hierarchy emerge.',
    seeAlso: ['morph', 'graph', 'coder']
  },
  {
    id: 'morph',
    title: 'Morph·merge',
    kind: 'concept',
    blurb:
      'BPE-style morphology: greedy merges over a weighted word list until stems and affixes pay for themselves.',
    seeAlso: ['morfessor', 'grammar']
  },
  {
    id: 'morfessor',
    title: 'Morph·split',
    kind: 'concept',
    blurb:
      'Morfessor-style morphology: start from whole words and recursively re-split — same objective as Morph·merge, opposite search.',
    seeAlso: ['morph']
  },
  {
    id: 'graph',
    title: 'Graph·SUBDUE',
    kind: 'concept',
    blurb:
      'SUBDUE substructure compression: collapse the recurring subgraph that saves the most bits, then do it again on the result.',
    seeAlso: ['grammar']
  },
  {
    id: 'coder',
    title: 'Coder',
    kind: 'concept',
    blurb:
      'An arithmetic coder turning probabilities into literal bits — including probabilities from the Mini·GPT as it trains.',
    seeAlso: ['llm', 'grammar']
  }
];

export const lensById = (id: string): LensMeta | undefined =>
  LENSES.find((l) => l.id === id);
