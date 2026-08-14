/**
 * MORPHOLOGY LENS — Morfessor-style word-list segmentation (agglomerative).
 *
 * Data: a *word list* — distinct word types, each with a corpus frequency.
 *       Every word is held as a sequence of integer morph ids.
 * Model: a lexicon of morphs. A morph is named by a rule  M_k -> (s_i, s_j)
 *        that joins two adjacent sub-morphs, exactly like the grammar lens —
 *        so morphs compose into a hierarchy (m -> (mor, ph)) and the surface
 *        string of any morph is the concatenation of its parts.
 *
 * A MOVE merges one adjacent morph PAIR wherever it occurs *inside a word*
 * (never across word boundaries), weighted by the word's frequency. Starting
 * from words spelled out as single characters (L(M) tiny, L(D|M) huge), greedy
 * MDL descent grows the lexicon: stems like "walk" and affixes like "ing" / "ed"
 * emerge because they recur across many words and so pay for their lexicon entry
 * many times over in the corpus.
 *
 * This is the *agglomerative* variant — RePair / BPE restricted to word-internal
 * pairs over a frequency-weighted vocabulary, scored by a two-part MDL cost
 * (lexicon + corpus). It is faithful to Morfessor in spirit (two-part MDL over a
 * morph lexicon) but not in mechanism: canonical Morfessor Baseline starts from
 * whole words and *recursively splits*, revisiting earlier cuts to escape local
 * optima. That recursive-split engine is the planned phase 2; see ROADMAP.
 *
 * The cost mirrors the grammar lens so the same two code models apply:
 *
 *   'uniform' — every morph reference costs log2(V) bits, V = lexicon size.
 *   'shannon' — every morph reference costs -log2(p) bits, p = its frequency
 *               across the weighted corpus + lexicon bodies (real entropy
 *               coding: "ing" is cheap, a one-off morph is dear).
 */

import type { CodeMode, CostBreakdown, MdlProblem, ScoredMove } from '../mdl/types';
import { fmt, surprisal, uniformBits } from '../mdl/format';
import {
  vocabSize,
  expandVisible,
  replaceDigram,
  token as symToken
} from '../mdl/symbols';

export type { CodeMode };

export interface MorphConfig {
  codeMode: CodeMode;
  /** Bits to spell one alphabet character when transmitting the lexicon.
   *  Constant across models, shown for honesty (does not affect which move wins). */
  charBits: number;
  /** Include the cost of transmitting the model-of-the-model (lexicon framing /
   *  code table). Turning this off isolates the "pure" two-part intuition. */
  includeOverhead: boolean;
}

/** A learned morph: right-hand side is exactly two sub-morph ids. */
export interface MorphRule {
  rhs: [number, number];
}

/** One word type in the corpus, with its current segmentation and frequency. */
export interface Word {
  /** Surface form, for display. */
  surface: string;
  /** Corpus frequency (token count) of this word type. */
  count: number;
  /** Current segmentation: sequence of morph ids. */
  seq: number[];
}

export interface MorphModel {
  /** Distinct base characters. terminal id === index here. */
  terminals: string[];
  /** Learned morphs. morph k has symbol id  terminals.length + k. */
  rules: MorphRule[];
  /** The data under the model: the word list, each word a morph-id sequence. */
  words: Word[];
  config: MorphConfig;
}

/** A move = "join this adjacent morph pair into a new lexicon entry". */
export interface MergeMove {
  a: number;
  b: number;
}

export const defaultConfig: MorphConfig = {
  codeMode: 'uniform',
  charBits: 8,
  includeOverhead: true
};

// ---------------------------------------------------------------------------
// Symbol helpers (morph ids: < terminals.length are chars, else learned
// morphs) — the shared algebra in ../mdl/symbols, identical to the string
// lens's. Re-exported so this lens's public API is unchanged.
// ---------------------------------------------------------------------------

export {
  vocabSize,
  isTerminal,
  ruleIndexOf,
  showChar,
  expand,
  expandVisible,
  replaceDigram
} from '../mdl/symbols';

/** Printable token for a morph id: the char itself, or "M{k}". */
export const token = (m: MorphModel, id: number): string => symToken(m, id, 'M');

// ---------------------------------------------------------------------------
// Digram counting — WORD-BOUNDED and FREQUENCY-WEIGHTED. This is where the
// morphology lens genuinely diverges from the string lens: same replaceDigram
// fold (applied per word in `apply`), different way of COUNTING occurrences.
// ---------------------------------------------------------------------------

const key = (a: number, b: number): string => `${a},${b}`;

/**
 * Count adjacent morph pairs across all words. Pairs never cross a word
 * boundary, and each occurrence is weighted by the word's corpus frequency —
 * that weighting is what makes this MDL care about how *often* a morph is used,
 * not just in how many distinct word types it appears.
 */
export function countDigrams(words: Word[]): Map<string, { a: number; b: number; n: number }> {
  const counts = new Map<string, { a: number; b: number; n: number }>();
  for (const w of words) {
    const seq = w.seq;
    let i = 0;
    while (i + 1 < seq.length) {
      const a = seq[i];
      const b = seq[i + 1];
      const k = key(a, b);
      const e = counts.get(k);
      if (e) e.n += w.count;
      else counts.set(k, { a, b, n: w.count });
      // Non-overlapping: "aaa" within a word counts its pair once.
      if (a === b && i + 2 < seq.length && seq[i + 2] === a) i += 2;
      else i += 1;
    }
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Frequency model (for Shannon coding) — weighted corpus tokens + morph bodies
// ---------------------------------------------------------------------------

function symbolCounts(m: MorphModel): Map<number, number> {
  const counts = new Map<number, number>();
  const bump = (id: number, w: number) => counts.set(id, (counts.get(id) ?? 0) + w);
  // Corpus: each morph token weighted by its word's frequency.
  for (const word of m.words) for (const id of word.seq) bump(id, word.count);
  // Lexicon bodies: each referenced sub-morph counts once (as in the grammar lens).
  for (const r of m.rules) {
    bump(r.rhs[0], 1);
    bump(r.rhs[1], 1);
  }
  return counts;
}

/** Total morph tokens emitted by the corpus, weighted by word frequency. */
const corpusTokens = (m: MorphModel): number =>
  m.words.reduce((s, w) => s + w.count * w.seq.length, 0);

// ---------------------------------------------------------------------------
// Cost: L(M) lexicon + L(D|M) corpus
// ---------------------------------------------------------------------------

export function cost(m: MorphModel): CostBreakdown {
  const cfg = m.config;
  const alphabetBits = m.terminals.length * cfg.charBits;
  const tokens = corpusTokens(m);

  if (cfg.codeMode === 'uniform') {
    // Uniform code, as in the grammar lens but over frequency-WEIGHTED tokens:
    //   L(D|M) = (Σ_w count(w)·|seq(w)|) · log2(V)       (corpus morph tokens)
    //   L(M)   = 2·|rules| · log2(V)                     (morph bodies, 2 refs each)
    //          + |rules| · log2(V)      [if overhead]    (per-entry framing slot)
    //          + |terminals| · charBits                  (spell the alphabet; fixed)
    // A merge seen n weighted times removes n tokens from the corpus term but
    // adds a lexicon entry and nudges log2(V) up for everyone — a morph earns
    // its keep only when n amortises that.
    const V = vocabSize(m);
    const bps = uniformBits(V); // bits per morph reference

    const dataBits = tokens * bps;
    const ruleBodyBits = m.rules.reduce((s, r) => s + r.rhs.length * bps, 0);
    // One framing slot per lexicon entry so the decoder knows morph boundaries.
    const framingBits = cfg.includeOverhead ? m.rules.length * bps : 0;

    const modelBits = ruleBodyBits + framingBits + alphabetBits;

    return {
      modelBits,
      dataBits,
      total: modelBits + dataBits,
      modelTerms: [
        {
          label: 'morph bodies',
          bits: ruleBodyBits,
          detail: `${m.rules.length} morphs × 2 refs × ${fmt(bps)} bits/ref`
        },
        ...(cfg.includeOverhead
          ? [{ label: 'lexicon framing', bits: framingBits, detail: `${m.rules.length} × ${fmt(bps)} bits` }]
          : []),
        {
          label: 'alphabet',
          bits: alphabetBits,
          detail: `${m.terminals.length} chars × ${cfg.charBits} bits`,
          fixed: true
        }
      ],
      dataTerms: [
        {
          label: 'corpus (morph tokens)',
          bits: dataBits,
          detail: `${tokens} weighted tokens × ${fmt(bps)} bits/token`
        }
      ],
      meta: {
        'bits/token': bps,
        lexicon: V,
        'corpus tokens': tokens,
        morphs: m.rules.length
      }
    };
  }

  // Shannon / entropy coding: pool weighted counts (corpus tokens × word
  // frequency, + 1 per rule-body ref) and charge each occurrence its ideal
  // codelength −log2(c_id / total). dataBits below is the per-word sum of morph
  // codelengths re-weighted by word frequency — algebraically the same as
  // Σ_id c_id·surprisal restricted to corpus occurrences.
  const counts = symbolCounts(m);
  let total = 0;
  for (const c of counts.values()) total += c;
  const codeLen = new Map<number, number>();
  for (const [id, c] of counts) codeLen.set(id, surprisal(c / total));

  const len = (id: number) => codeLen.get(id) ?? 0;
  const dataBits = m.words.reduce(
    (s, w) => s + w.count * w.seq.reduce((t, id) => t + len(id), 0),
    0
  );
  const ruleBodyBits = m.rules.reduce((s, r) => s + len(r.rhs[0]) + len(r.rhs[1]), 0);
  // Transmit the code table: one frequency per distinct morph, each a count in
  // [0..total] sent with a uniform code of log2(total+1) bits.
  const codeTableBits = cfg.includeOverhead ? counts.size * uniformBits(total + 1) : 0;

  const modelBits = ruleBodyBits + codeTableBits + alphabetBits;
  const bps = tokens > 0 ? dataBits / tokens : 0;

  return {
    modelBits,
    dataBits,
    total: modelBits + dataBits,
    modelTerms: [
      {
        label: 'morph bodies',
        bits: ruleBodyBits,
        detail: `${m.rules.length} morphs, entropy-coded refs`
      },
      ...(cfg.includeOverhead
        ? [{ label: 'code table', bits: codeTableBits, detail: `${counts.size} morphs × ${fmt(uniformBits(total + 1))} bits` }]
        : []),
      {
        label: 'alphabet',
        bits: alphabetBits,
        detail: `${m.terminals.length} chars × ${cfg.charBits} bits`,
        fixed: true
      }
    ],
    dataTerms: [
      {
        label: 'corpus (morph tokens)',
        bits: dataBits,
        detail: `${tokens} weighted tokens, entropy-coded (${fmt(bps)} avg bits/token)`
      }
    ],
    meta: {
      'avg bits/token': bps,
      lexicon: vocabSize(m),
      'corpus tokens': tokens,
      morphs: m.rules.length
    }
  };
}

// ---------------------------------------------------------------------------
// Candidate moves + scoring
// ---------------------------------------------------------------------------

export function candidates(m: MorphModel): MergeMove[] {
  const counts = countDigrams(m.words);
  const out: MergeMove[] = [];
  for (const { a, b, n } of counts.values()) {
    // Merge condition: weighted count ≥ 2. n is frequency-weighted, so a pair
    // inside a single word type of count 2 qualifies; a pair occurring once in
    // a count-1 word cannot repay its 2-ref lexicon body.
    if (n >= 2) out.push({ a, b });
  }
  return out;
}

export function apply(m: MorphModel, move: MergeMove): MorphModel {
  const newId = vocabSize(m);
  const rules = [...m.rules, { rhs: [move.a, move.b] as [number, number] }];
  // Shared replaceDigram, applied per word — merges never cross a word boundary.
  const words = m.words.map((w) => ({
    ...w,
    seq: replaceDigram(w.seq, move.a, move.b, newId)
  }));
  return { ...m, rules, words };
}

export function scoreMove(
  m: MorphModel,
  move: MergeMove,
  baseline: CostBreakdown
): ScoredMove<MergeMove> {
  // Exact ΔL by full re-cost (delta = total′ − total), as in the grammar lens.
  // The difference here: n is the frequency-WEIGHTED occurrence count, so a
  // merge inside one very common word can beat one spread across many rare
  // types — that is what pulls out corpus-frequent stems and affixes.
  const after = cost(apply(m, move));
  const counts = countDigrams(m.words);
  const n = counts.get(`${move.a},${move.b}`)?.n ?? 0;
  const nextIdx = m.rules.length;
  const expansion = expandVisible(m, move.a) + expandVisible(m, move.b);

  return {
    move,
    label: `${token(m, move.a)}·${token(m, move.b)} → M${nextIdx}`,
    delta: after.total - baseline.total,
    totalAfter: after.total,
    modelBitsAfter: after.modelBits,
    dataBitsAfter: after.dataBits,
    extra: {
      '×': n,
      expands: `“${expansion}”`
    }
  };
}

// ---------------------------------------------------------------------------
// Word-list parsing + problem factory
// ---------------------------------------------------------------------------

/**
 * Parse a word list. One word per line; an optional integer after whitespace is
 * its corpus frequency ("walking 12"). A line of bare space-separated words
 * gives each a count of 1. Duplicate words aggregate. This is the *data* D.
 */
export function parseWordList(text: string): { surface: string; count: number }[] {
  const counts = new Map<string, number>();
  const order: string[] = [];
  const bump = (word: string, n: number) => {
    if (!counts.has(word)) order.push(word);
    counts.set(word, (counts.get(word) ?? 0) + n);
  };

  for (const line of text.split('\n')) {
    const toks = line.trim().split(/\s+/).filter(Boolean);
    if (toks.length === 0) continue;
    if (toks.length === 2 && /^\d+$/.test(toks[1])) {
      bump(toks[0], parseInt(toks[1], 10));
    } else {
      for (const t of toks) if (!/^\d+$/.test(t)) bump(t, 1);
    }
  }

  return order.map((surface) => ({ surface, count: counts.get(surface) ?? 1 }));
}

export function buildInitialModel(text: string, config: MorphConfig): MorphModel {
  const terminals: string[] = [];
  const index = new Map<string, number>();
  const charId = (ch: string): number => {
    let id = index.get(ch);
    if (id === undefined) {
      id = terminals.length;
      terminals.push(ch);
      index.set(ch, id);
    }
    return id;
  };

  const words: Word[] = parseWordList(text).map(({ surface, count }) => ({
    surface,
    count,
    seq: [...surface].map(charId)
  }));

  return { terminals, rules: [], words, config };
}

export function morphologyProblem(
  text: string,
  config: MorphConfig
): MdlProblem<MorphModel, MergeMove> {
  const initial = buildInitialModel(text, config);
  return {
    name: 'Morphology segmentation',
    blurb:
      'Segment a frequency-weighted word list. Greedily join recurring morph pairs into a lexicon and watch stems and affixes emerge as the lexicon (model) trades off against the segmented corpus (data).',
    initialModel: () => initial,
    cost,
    candidates,
    apply,
    scoreMove
  };
}
