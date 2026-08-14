/**
 * Shared symbol algebra for the digram-merging lenses (string grammar and
 * morphology).
 *
 * Both lenses build the SAME kind of model: an integer symbol space where ids
 * below |terminals| are base characters, and every id above that names a
 * learned rule  X_k -> (s_i, s_j)  joining two earlier symbols — so learned
 * symbols compose into a hierarchy, and the surface string of any symbol is
 * the concatenation of its parts. That shared shape is `SymbolSpace`, and
 * everything in this file operates on it.
 *
 * What the lenses do NOT share stays in the lens files: what a rule means
 * (grammar rule over one stream vs. morph inside words), how occurrences are
 * counted (raw adjacency vs. word-bounded and frequency-weighted), and the
 * cost model. One algorithm, different counting — this file is the "one
 * algorithm" half made structural.
 */

/** The minimal model shape the shared algebra needs. */
export interface SymbolSpace {
  /** Distinct base characters. terminal id === index here. */
  terminals: string[];
  /** Learned rules. rule k has symbol id  terminals.length + k. */
  rules: { rhs: [number, number] }[];
}

// ---------------------------------------------------------------------------
// Symbol helpers
// ---------------------------------------------------------------------------

export const vocabSize = (m: SymbolSpace): number =>
  m.terminals.length + m.rules.length;

export const isTerminal = (m: SymbolSpace, id: number): boolean =>
  id < m.terminals.length;

export const ruleIndexOf = (m: SymbolSpace, id: number): number =>
  id - m.terminals.length;

/** Printable token for a symbol id: the char itself, or "{prefix}{k}" — each
 *  lens picks its prefix ('R' for grammar rules, 'M' for morphs). */
export function token(m: SymbolSpace, id: number, prefix: string): string {
  if (isTerminal(m, id)) return showChar(m.terminals[id]);
  return `${prefix}${ruleIndexOf(m, id)}`;
}

/** Render a single character visibly (whitespace made explicit). */
export function showChar(ch: string): string {
  if (ch === ' ') return '␣';
  if (ch === '\n') return '⏎';
  if (ch === '\t') return '⇥';
  return ch;
}

/** Fully expand a symbol id to its underlying terminal string. */
export function expand(m: SymbolSpace, id: number, memo = new Map<number, string>()): string {
  if (isTerminal(m, id)) return m.terminals[id];
  const cached = memo.get(id);
  if (cached !== undefined) return cached;
  const rule = m.rules[ruleIndexOf(m, id)];
  const s = expand(m, rule.rhs[0], memo) + expand(m, rule.rhs[1], memo);
  memo.set(id, s);
  return s;
}

/** Visible expansion (whitespace made explicit) for display. */
export const expandVisible = (m: SymbolSpace, id: number): string =>
  [...expand(m, id)].map(showChar).join('');

// ---------------------------------------------------------------------------
// Digram replacement (non-overlapping, left-to-right)
// ---------------------------------------------------------------------------

/** Replace every non-overlapping occurrence of digram (a,b) with newId.
 *  Greedy left-to-right — each lens's digram COUNTER scans the same way, so a
 *  digram counted n times shrinks its sequence by exactly n symbols
 *  (each occurrence 2 → 1). Grammar folds one long stream; morphology applies
 *  this per word, so merges never cross a word boundary. */
export function replaceDigram(seq: number[], a: number, b: number, newId: number): number[] {
  const out: number[] = [];
  let i = 0;
  while (i < seq.length) {
    if (i + 1 < seq.length && seq[i] === a && seq[i + 1] === b) {
      out.push(newId);
      i += 2;
    } else {
      out.push(seq[i]);
      i += 1;
    }
  }
  return out;
}
