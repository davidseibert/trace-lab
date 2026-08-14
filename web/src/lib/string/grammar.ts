/**
 * STRING LENS — grammar / dictionary compression (RePair-style).
 *
 * Data: a string over some alphabet, held as a sequence of integer symbol ids.
 * Model: a set of rewrite rules  R_k -> (s_i, s_j)  that name a recurring pair
 *        of symbols. Each rule introduces a new nonterminal symbol.
 *
 * A MOVE introduces one rule for an adjacent digram and replaces every
 * (non-overlapping) occurrence of that digram in the sequence with the new
 * symbol. Over successive steps the rules compose — R0 -> (t,h), then
 * R1 -> (R0,e) — so longer structures like "the" emerge as a hierarchy. That
 * hierarchy IS the induced grammar.
 *
 * Two code models are provided, toggled by `codeMode`:
 *
 *   'uniform' — every symbol reference costs log2(V) bits, where V is the
 *               current vocabulary size. Dead simple to reason about. Adding a
 *               rule grows V (each symbol gets slightly pricier) but shrinks the
 *               sequence (far fewer symbols). That tension is the MDL story in
 *               its most legible form.
 *
 *   'shannon' — every symbol costs -log2(p) bits, where p is its empirical
 *               frequency across the whole message (sequence + rule bodies).
 *               This is real entropy coding: frequent symbols get short codes.
 *               Closer to what actual compressors achieve.
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

export interface GrammarConfig {
  codeMode: CodeMode;
  /** Bits to spell one alphabet character when transmitting the dictionary.
   *  Constant across models, shown for honesty (does not affect which move wins). */
  charBits: number;
  /** Include the cost of transmitting the model-of-the-model (code table /
   *  rule framing). Turning this off isolates the "pure" two-part intuition. */
  includeOverhead: boolean;
}

export interface Rule {
  /** Right-hand side: exactly two symbol ids (a digram). */
  rhs: [number, number];
}

export interface GrammarModel {
  /** Distinct base characters. terminal id === index here. */
  terminals: string[];
  /** Learned rules. rule k has symbol id  terminals.length + k. */
  rules: Rule[];
  /** The data under the model: current compressed sequence of symbol ids. */
  sequence: number[];
  config: GrammarConfig;
}

/** A move = "create a rule for this digram and fold it in". */
export interface DigramMove {
  a: number;
  b: number;
}

export const defaultConfig: GrammarConfig = {
  codeMode: 'uniform',
  charBits: 8,
  includeOverhead: true
};

// ---------------------------------------------------------------------------
// Symbol helpers — the shared algebra in ../mdl/symbols, identical to the
// morphology lens's (same integer symbol space, same rule shape). Re-exported
// so this lens's public API is unchanged.
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

/** Printable token for a symbol id: the char itself, or "R{k}". */
export const token = (m: GrammarModel, id: number): string => symToken(m, id, 'R');

// ---------------------------------------------------------------------------
// Digram counting (non-overlapping, left-to-right)
// ---------------------------------------------------------------------------

const key = (a: number, b: number): string => `${a},${b}`;

/** Count non-overlapping occurrences of every adjacent digram.
 *  Non-overlapping counting matters for the math: replaceDigram scans the same
 *  greedy left-to-right way, so a digram counted n times shrinks the sequence
 *  by exactly n symbols (each occurrence 2 → 1). */
export function countDigrams(seq: number[]): Map<string, { a: number; b: number; n: number }> {
  const counts = new Map<string, { a: number; b: number; n: number }>();
  let i = 0;
  while (i + 1 < seq.length) {
    const a = seq[i];
    const b = seq[i + 1];
    const k = key(a, b);
    const e = counts.get(k);
    if (e) e.n++;
    else counts.set(k, { a, b, n: 1 });
    // Non-overlapping: if we matched, skip the pair so "aaa" counts once.
    if (a === b && i + 2 < seq.length && seq[i + 2] === a) {
      i += 2;
    } else {
      i += 1;
    }
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Frequency model (for Shannon coding) — counts over sequence + rule bodies
// ---------------------------------------------------------------------------

function symbolCounts(m: GrammarModel): Map<number, number> {
  const counts = new Map<number, number>();
  const bump = (id: number) => counts.set(id, (counts.get(id) ?? 0) + 1);
  for (const id of m.sequence) bump(id);
  for (const r of m.rules) {
    bump(r.rhs[0]);
    bump(r.rhs[1]);
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Cost: L(M) + L(D | M)
// ---------------------------------------------------------------------------

export function cost(m: GrammarModel): CostBreakdown {
  const cfg = m.config;
  const alphabetBits = m.terminals.length * cfg.charBits;

  if (cfg.codeMode === 'uniform') {
    // Uniform code: every symbol reference — in the stream or in a rule body —
    // costs the same log2(V) bits, V = |terminals| + |rules|. So
    //   L(D|M) = |sequence| · log2(V)                    (compressed stream)
    //   L(M)   = 2·|rules| · log2(V)                     (rule bodies, 2 refs each)
    //          + |rules| · log2(V)      [if overhead]    (per-rule framing slot)
    //          + |terminals| · charBits                  (spell the alphabet; fixed)
    // Adding a rule raises log2(V) for EVERY reference in the message — that is
    // the price an n-fold replacement's savings must beat.
    const V = vocabSize(m);
    const bps = uniformBits(V); // bits per symbol reference

    const dataBits = m.sequence.length * bps;
    const ruleBodyBits = m.rules.reduce((s, r) => s + r.rhs.length * bps, 0);
    // One length/terminator slot per rule so the decoder knows rule boundaries.
    const framingBits = cfg.includeOverhead ? m.rules.length * bps : 0;

    const modelBits = ruleBodyBits + framingBits + alphabetBits;

    return {
      modelBits,
      dataBits,
      total: modelBits + dataBits,
      modelTerms: [
        {
          label: 'rule bodies',
          bits: ruleBodyBits,
          detail: `${m.rules.length} rules × 2 symbols × ${fmt(bps)} bits/symbol`
        },
        ...(cfg.includeOverhead
          ? [{ label: 'rule framing', bits: framingBits, detail: `${m.rules.length} × ${fmt(bps)} bits` }]
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
          label: 'compressed stream',
          bits: dataBits,
          detail: `${m.sequence.length} symbols × ${fmt(bps)} bits/symbol`
        }
      ],
      meta: {
        'bits/symbol': bps,
        vocabulary: V,
        'stream length': m.sequence.length,
        rules: m.rules.length
      }
    };
  }

  // Shannon / entropy coding: pool occurrence counts over the whole message
  // (sequence + rule bodies), then charge each occurrence of symbol i its ideal
  // codelength −log2(c_i / total). Summed, L = Σ_i c_i · −log2(c_i/total)
  // = total · H(empirical) — idealised entropy coding, no integer-length
  // rounding (real Huffman/arithmetic codes come within a bit of this).
  const counts = symbolCounts(m);
  let total = 0;
  for (const c of counts.values()) total += c;
  const codeLen = new Map<number, number>();
  for (const [id, c] of counts) codeLen.set(id, surprisal(c / total));

  const len = (id: number) => codeLen.get(id) ?? 0;
  const dataBits = m.sequence.reduce((s, id) => s + len(id), 0);
  const ruleBodyBits = m.rules.reduce((s, r) => s + len(r.rhs[0]) + len(r.rhs[1]), 0);
  // Transmit the code table: one frequency per distinct symbol, each a count in
  // [0..total] sent with a uniform code of log2(total+1) bits. Without this the
  // decoder could not rebuild the codelengths.
  const codeTableBits = cfg.includeOverhead ? counts.size * uniformBits(total + 1) : 0;

  const modelBits = ruleBodyBits + codeTableBits + alphabetBits;
  const bps = m.sequence.length > 0 ? dataBits / m.sequence.length : 0;

  return {
    modelBits,
    dataBits,
    total: modelBits + dataBits,
    modelTerms: [
      {
        label: 'rule bodies',
        bits: ruleBodyBits,
        detail: `${m.rules.length} rules, entropy-coded symbols`
      },
      ...(cfg.includeOverhead
        ? [{ label: 'code table', bits: codeTableBits, detail: `${counts.size} symbols × ${fmt(uniformBits(total + 1))} bits` }]
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
        label: 'compressed stream',
        bits: dataBits,
        detail: `${m.sequence.length} symbols, entropy-coded (${fmt(bps)} avg bits/symbol)`
      }
    ],
    meta: {
      'avg bits/symbol': bps,
      vocabulary: vocabSize(m),
      'stream length': m.sequence.length,
      rules: m.rules.length
    }
  };
}

// ---------------------------------------------------------------------------
// Candidate moves + scoring
// ---------------------------------------------------------------------------

export function candidates(m: GrammarModel): DigramMove[] {
  const counts = countDigrams(m.sequence);
  const out: DigramMove[] = [];
  for (const { a, b, n } of counts.values()) {
    // RePair condition: only digrams seen ≥ 2 times. A rule used once can never
    // pay — its body alone (2 refs) outweighs the single symbol it removes.
    if (n >= 2) out.push({ a, b });
  }
  return out;
}

export function apply(m: GrammarModel, move: DigramMove): GrammarModel {
  const newId = vocabSize(m);
  const rules = [...m.rules, { rhs: [move.a, move.b] as [number, number] }];
  const sequence = replaceDigram(m.sequence, move.a, move.b, newId);
  return { ...m, rules, sequence };
}

export function scoreMove(
  m: GrammarModel,
  move: DigramMove,
  baseline: CostBreakdown
): ScoredMove<DigramMove> {
  // ΔL is exact, by re-costing the full model after the move:
  //   delta = [L(M′) + L(D|M′)] − [L(M) + L(D|M)]
  // Under 'uniform' the trade it captures: the stream loses n symbols, the
  // model gains 2 refs (+ framing), and every surviving reference now pays
  // log2(V+1) instead of log2(V). Under 'shannon' all codelengths shift with
  // the new frequency profile. No closed-form ΔL shortcut — exactness over
  // speed, since the point is to study the numbers.
  const after = cost(apply(m, move));
  const counts = countDigrams(m.sequence);
  const n = counts.get(`${move.a},${move.b}`)?.n ?? 0;
  const nextIdx = m.rules.length;
  const expansion =
    expandVisible(m, move.a) + expandVisible(m, move.b);

  return {
    move,
    label: `${token(m, move.a)}·${token(m, move.b)} → R${nextIdx}`,
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
// Problem factory
// ---------------------------------------------------------------------------

export function buildInitialModel(text: string, config: GrammarConfig): GrammarModel {
  const terminals: string[] = [];
  const index = new Map<string, number>();
  const sequence: number[] = [];
  for (const ch of text) {
    let id = index.get(ch);
    if (id === undefined) {
      id = terminals.length;
      terminals.push(ch);
      index.set(ch, id);
    }
    sequence.push(id);
  }
  return { terminals, rules: [], sequence, config };
}

export function grammarProblem(
  text: string,
  config: GrammarConfig
): MdlProblem<GrammarModel, DigramMove> {
  const initial = buildInitialModel(text, config);
  return {
    name: 'Grammar compression',
    blurb:
      'Find recurring digrams, name them with rules, and watch the dictionary (model) trade off against the compressed stream (data).',
    initialModel: () => initial,
    cost,
    candidates,
    apply,
    scoreMove
  };
}
