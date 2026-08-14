/**
 * MORFESSOR LENS — recursive-split word-list segmentation (Morfessor Baseline).
 *
 * Same data as the agglomerative morphology lens — a frequency-weighted word
 * list — optimized from the OPPOSITE extreme. Every word starts as a single
 * whole-word morph (lexicon huge: each word spelled out in the lexicon; corpus
 * tiny: one token per word). The algorithm then re-segments words to share
 * structure, sweeping the list in epochs until a pass changes nothing.
 *
 * Why a separate engine (not `MdlProblem` + `trace()`):
 *
 *   - A "move" here is RE-SEGMENTING ONE WORD, and its score depends on the
 *     whole model: the corpus code is -log2(p) over morph tokens, so a word's
 *     best cut is relative to every other word's counts. We therefore maintain
 *     incremental morph counts and re-optimize one word at a time.
 *   - It loops in EPOCHS, not one global argmin. Because "keep the current cut"
 *     is always a candidate, each re-analysis is non-increasing — a clean
 *     descending staircase that still shows RE-ANALYSIS: a word can flip its cut
 *     in a later epoch because other words moved the counts beneath it. We stop
 *     when a full pass changes nothing (a local optimum).
 *
 * Words are short, so each re-analysis enumerates EVERY segmentation and scores
 * it by the exact total cost — correct by construction, no approximation. (The
 * classic recursive O(n^2) splitter is just an optimization we don't need here;
 * for the rare long word we fall back to a bounded candidate set.)
 *
 * The cost is the two-part Morfessor MDL cost, with the same uniform/Shannon
 * toggle as the other lenses (the toggle changes the CORPUS code; the lexicon is
 * always char-spelled, since the lexicon literally stores morph strings):
 *
 *   lexicon  L(M)   = sum over distinct morphs of (len+1) * log2(alphabet+1)
 *                     (+1 char = a morph-boundary symbol)
 *   corpus   L(D|M) = uniform: tokens * log2(V)       V = lexicon size
 *                     shannon: sum over tokens of -log2(p)   (real entropy code)
 */

import type { CodeMode, CostBreakdown } from '../mdl/types';
import { fmt, surprisal, uniformBits } from '../mdl/format';
import { parseWordList } from '../morphology/morphology';

export type { CodeMode };

export interface MorfConfig {
  codeMode: CodeMode;
  /** Bits to declare one alphabet character exists. Constant, shown for honesty. */
  charBits: number;
}

export interface MorfWord {
  surface: string;
  count: number;
}

/** An immutable snapshot of the segmentation state — the "model" at a step. */
export interface MorfState {
  words: MorfWord[];
  /** analyses[i] = the morph strings word i is currently segmented into. */
  analyses: string[][];
  alphabetSize: number;
  config: MorfConfig;
}

/** One scored candidate segmentation of the focus word. */
export interface SegCandidate {
  seg: string[];
  /** Total bits of the whole model with the focus word cut this way. */
  total: number;
  /** total − current total. Negative = improvement. */
  delta: number;
}

/** One snapshot in the Morfessor trace — everything the UI needs to render. */
export interface MorfStep {
  index: number;
  /** Epoch number, or -1 for the terminal converged marker. */
  epoch: number;
  /** Model state AT this step (before this word's re-analysis is applied). */
  model: MorfState;
  /** Cost of this model state. */
  cost: CostBreakdown;
  /** Word being re-analysed at this step (null at the converged marker). */
  focusWord: number | null;
  /** The focus word's segmentation before re-analysis. */
  oldSeg: string[];
  /** Candidate segmentations of the focus word, best-first (capped). */
  candidates: SegCandidate[];
  /** The segmentation chosen (applied) — null at the converged marker. */
  chosen: SegCandidate | null;
  /** Did the chosen segmentation differ from the old one? */
  changed: boolean;
  /** Narration of what happens at this step. */
  note: string;
}

export const defaultConfig: MorfConfig = {
  codeMode: 'shannon', // -log2(p) corpus code is the distinctive Morfessor knob
  charBits: 8
};

// ---------------------------------------------------------------------------
// Morph counts + cost
// ---------------------------------------------------------------------------

/** Weighted morph-token counts implied by a set of analyses. */
export function countsOf(words: MorfWord[], analyses: string[][]): Map<string, number> {
  const c = new Map<string, number>();
  for (let i = 0; i < words.length; i++) {
    const w = words[i].count;
    for (const m of analyses[i]) c.set(m, (c.get(m) ?? 0) + w);
  }
  return c;
}

// Incremental bookkeeping for leave-one-out scoring. removeSeg deletes a morph
// outright when its count reaches 0 — that matters, because the lexicon term in
// costComponents is keyed on counts.keys(): a morph's spelling bits (and its
// slot in V) vanish the moment its last token does.
const addSeg = (counts: Map<string, number>, seg: string[], w: number) => {
  for (const m of seg) counts.set(m, (counts.get(m) ?? 0) + w);
};

const removeSeg = (counts: Map<string, number>, seg: string[], w: number) => {
  for (const m of seg) {
    const v = (counts.get(m) ?? 0) - w;
    if (v <= 0) counts.delete(m);
    else counts.set(m, v);
  }
};

/** The itemised components of the Morfessor cost — the ONE place the
 *  arithmetic lives. Both the hot search loop (`totalBits`) and the itemised
 *  display (`cost`) call this, so what the search optimizes and what the UI
 *  shows cannot drift apart.
 *
 *  total = L(M) + L(D|M), with T = Σ counts (corpus tokens), V = |lexicon|:
 *    L(M)   = Σ_{m ∈ lexicon} (|m|+1) · log2(A+1)   — spell each morph char by
 *             char over the A-letter alphabet + an end-of-morph symbol (the +1s)
 *           + A · charBits                          — declare the alphabet (fixed)
 *    L(D|M) = uniform:  T · log2(V)
 *             shannon:  Σ_types c · (−log2(c/T))    — per-token −log2(p); tokens
 *                       of a type share a codelength, so we sum by type
 *  The lexicon is whatever has ≥1 token, so re-cutting a word away from a morph
 *  automatically shrinks both the spelling sum and V. */
function costComponents(
  counts: Map<string, number>,
  alphabetSize: number,
  cfg: MorfConfig
): { T: number; V: number; perSym: number; lexiconBits: number; alphabetBits: number; dataBits: number } {
  let T = 0;
  for (const v of counts.values()) T += v;
  const V = counts.size;

  const perSym = uniformBits(alphabetSize + 1);
  let lexiconBits = 0;
  for (const m of counts.keys()) lexiconBits += (m.length + 1) * perSym;
  const alphabetBits = alphabetSize * cfg.charBits;

  let dataBits: number;
  if (cfg.codeMode === 'uniform') {
    dataBits = T * uniformBits(V);
  } else {
    let s = 0;
    for (const v of counts.values()) s += v * surprisal(v / T);
    dataBits = s;
  }

  return { T, V, perSym, lexiconBits, alphabetBits, dataBits };
}

/** Fast total bits for the hot scoring loop (no itemised breakdown). */
function totalBits(counts: Map<string, number>, alphabetSize: number, cfg: MorfConfig): number {
  const c = costComponents(counts, alphabetSize, cfg);
  return c.lexiconBits + c.alphabetBits + c.dataBits;
}

/** Full itemised cost for a state snapshot. Delegates to `costComponents` —
 *  the same components the search's `totalBits` sums — and only attaches
 *  labels for the UI, so search and display share one set of formulas by
 *  construction. */
export function cost(state: MorfState): CostBreakdown {
  const counts = countsOf(state.words, state.analyses);
  const cfg = state.config;
  const A = state.alphabetSize;

  const { T, V, perSym, lexiconBits, alphabetBits, dataBits } = costComponents(counts, A, cfg);

  const modelBits = lexiconBits + alphabetBits;
  const bps = T > 0 ? dataBits / T : 0;

  return {
    modelBits,
    dataBits,
    total: modelBits + dataBits,
    modelTerms: [
      {
        label: 'morph spellings',
        bits: lexiconBits,
        detail: `${V} morphs × (len+1) × ${fmt(perSym)} bits/char`
      },
      {
        label: 'alphabet',
        bits: alphabetBits,
        detail: `${A} chars × ${cfg.charBits} bits`,
        fixed: true
      }
    ],
    dataTerms: [
      {
        label: 'corpus (morph tokens)',
        bits: dataBits,
        detail:
          cfg.codeMode === 'uniform'
            ? `${T} tokens × ${fmt(uniformBits(V))} bits/token`
            : `${T} tokens, entropy-coded (${fmt(bps)} avg bits/token)`
      }
    ],
    meta: {
      'avg bits/token': bps,
      'morph types': V,
      'corpus tokens': T
    }
  };
}

// ---------------------------------------------------------------------------
// Candidate segmentations
// ---------------------------------------------------------------------------

/** Cap on cut points we enumerate exhaustively (2^cap segmentations). */
const MAX_CUTS = 12;

/**
 * All segmentations of a string. For a string of length n there are n−1 cut
 * points and 2^(n−1) segmentations — exhaustive, so the chosen cut is exactly
 * cost-minimal. For the rare long word, fall back to a bounded set: the whole
 * word plus every single two-way split.
 */
export function candidateSegs(s: string): string[][] {
  const cuts = s.length - 1;
  if (cuts <= 0) return [[s]];

  if (cuts > MAX_CUTS) {
    const out: string[][] = [[s]];
    for (let k = 1; k < s.length; k++) out.push([s.slice(0, k), s.slice(k)]);
    return out;
  }

  // Bit k of mask = "cut after character k", so each mask in [0, 2^(n−1)) is
  // one distinct segmentation; mask 0 is the whole word.
  const out: string[][] = [];
  for (let mask = 0; mask < 1 << cuts; mask++) {
    const seg: string[] = [];
    let start = 0;
    for (let k = 0; k < cuts; k++) {
      if (mask & (1 << k)) {
        seg.push(s.slice(start, k + 1));
        start = k + 1;
      }
    }
    seg.push(s.slice(start));
    out.push(seg);
  }
  return out;
}

const sameSeg = (a: string[], b: string[]): boolean =>
  a.length === b.length && a.every((m, i) => m === b[i]);

// ---------------------------------------------------------------------------
// The engine: epoch sweeps of exact per-word re-analysis
// ---------------------------------------------------------------------------

export interface MorfTraceOptions {
  maxEpochs?: number;
  maxSteps?: number;
}

export function morfessorTrace(
  words: MorfWord[],
  alphabetSize: number,
  config: MorfConfig,
  opts: MorfTraceOptions = {}
): MorfStep[] {
  const maxEpochs = opts.maxEpochs ?? 30;
  const maxSteps = opts.maxSteps ?? 2000;

  // Init: every word is one whole-word morph (lexicon-heavy extreme).
  const analyses: string[][] = words.map((w) => [w.surface]);
  const counts = countsOf(words, analyses);

  const steps: MorfStep[] = [];
  let idx = 0;
  let converged = false;

  for (let epoch = 0; epoch < maxEpochs && !converged && idx < maxSteps; epoch++) {
    let changedAny = false;

    for (let i = 0; i < words.length && idx < maxSteps; i++) {
      const w = words[i];
      const oldSeg = analyses[i];
      const curTotal = totalBits(counts, alphabetSize, config);

      // Snapshot the pre-decision state (word i still in its old segmentation).
      const snapAnalyses = analyses.map((a) => a.slice());
      const stepCost = cost({ words, analyses: snapAnalyses, alphabetSize, config });

      // Pull word i out, score every candidate against the rest of the model —
      // leave-one-out re-estimation, the coordinate-descent heart of Morfessor.
      // Each candidate's `total` is the EXACT global cost with that cut in
      // place: under 'shannon' its morphs' codelengths −log2(c/T) already
      // include the candidate's own contribution to the counts, so sharing a
      // morph with other words genuinely cheapens it. delta = total − curTotal;
      // re-adding oldSeg reproduces curTotal exactly, so "keep" always scores
      // delta = 0 and best.total ≤ curTotal — the descent cannot go uphill.
      removeSeg(counts, oldSeg, w.count);
      const scored: SegCandidate[] = candidateSegs(w.surface).map((seg) => {
        addSeg(counts, seg, w.count);
        const total = totalBits(counts, alphabetSize, config);
        removeSeg(counts, seg, w.count);
        return { seg, total, delta: total - curTotal };
      });
      scored.sort(
        (a, b) =>
          a.total - b.total ||
          a.seg.length - b.seg.length ||
          a.seg.join('').localeCompare(b.seg.join(''))
      );

      const best = scored[0];
      // Only move on a strict improvement (1e-9 absorbs float noise), so the
      // descent is clean and monotone and ties never cause flip-flopping —
      // which is also what makes "a full pass changed nothing" a sound
      // convergence test.
      const changed = best.total < curTotal - 1e-9 && !sameSeg(best.seg, oldSeg);
      const applied = changed ? best.seg : oldSeg;
      addSeg(counts, applied, w.count);
      analyses[i] = applied;
      if (changed) changedAny = true;

      const chosen =
        scored.find((s) => sameSeg(s.seg, applied)) ??
        ({ seg: applied, total: curTotal, delta: 0 } as SegCandidate);

      steps.push({
        index: idx,
        epoch,
        model: { words, analyses: snapAnalyses, alphabetSize, config },
        cost: stepCost,
        focusWord: i,
        oldSeg,
        candidates: scored.slice(0, 12),
        chosen,
        changed,
        note: changed
          ? `Re-analyse “${w.surface}”: ${oldSeg.join('·')} → ${best.seg.join('·')} — saves ${(curTotal - best.total).toFixed(2)} bits.`
          : `Keep “${w.surface}” = ${oldSeg.join('·')} — no cut lowers the total.`
      });
      idx++;
    }

    if (!changedAny) converged = true;
  }

  // Terminal marker.
  const finalAnalyses = analyses.map((a) => a.slice());
  steps.push({
    index: idx,
    epoch: -1,
    model: { words, analyses: finalAnalyses, alphabetSize, config },
    cost: cost({ words, analyses: finalAnalyses, alphabetSize, config }),
    focusWord: null,
    oldSeg: [],
    candidates: [],
    chosen: null,
    changed: false,
    note: converged
      ? 'A full pass changed nothing. MDL local optimum — converged.'
      : 'Step budget reached — stopping.'
  });

  return steps;
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/** Distinct-character count of a word list (the alphabet size). */
export function alphabetOf(words: MorfWord[]): number {
  const set = new Set<string>();
  for (const w of words) for (const ch of w.surface) set.add(ch);
  return set.size;
}

/** Parse a word list (shared with the merge lens) into Morfessor words. */
export function buildWords(text: string): MorfWord[] {
  return parseWordList(text);
}

/** Visible expansion of a morph (whitespace made explicit) for display. */
export function showMorph(m: string): string {
  return [...m].map((ch) => (ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch === '\t' ? '⇥' : ch)).join('');
}

/** Per-morph lexicon rows for the lexicon view: morph, token count, spelling bits. */
export function lexiconRows(
  state: MorfState
): { morph: string; count: number; bits: number }[] {
  const counts = countsOf(state.words, state.analyses);
  const perSym = uniformBits(state.alphabetSize + 1);
  const rows = [...counts.entries()].map(([morph, count]) => ({
    morph,
    count,
    bits: (morph.length + 1) * perSym
  }));
  // Frequent, then longer (costlier) morphs first.
  rows.sort((a, b) => b.count - a.count || b.morph.length - a.morph.length || a.morph.localeCompare(b.morph));
  return rows;
}
