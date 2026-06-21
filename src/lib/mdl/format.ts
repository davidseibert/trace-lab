/** Tiny formatting + information-theory helpers shared across lenses. */

/** log base 2. */
export const log2 = (x: number): number => Math.log(x) / Math.LN2;

/**
 * Bits to choose one item out of `n` equally-likely items: log2(n).
 * Guards the degenerate cases (n <= 1 needs 0 bits — there is no choice).
 */
export const uniformBits = (n: number): number => (n <= 1 ? 0 : log2(n));

/**
 * Shannon information content of an event with probability p: -log2(p) bits.
 * This is the optimal code length for a symbol that occurs with frequency p.
 */
export const surprisal = (p: number): number => (p <= 0 ? 0 : -log2(p));

/**
 * Shannon entropy (bits/symbol) of a frequency map — the theoretical floor on
 * average code length. Sum of count[s]/T * surprisal(count[s]/T).
 */
export function entropy(counts: Iterable<number>): number {
  let total = 0;
  const arr: number[] = [];
  for (const c of counts) {
    if (c > 0) {
      arr.push(c);
      total += c;
    }
  }
  if (total === 0) return 0;
  let h = 0;
  for (const c of arr) {
    const p = c / total;
    h += p * surprisal(p);
  }
  return h;
}

/** Format a bit count for display: "1,234.5 bits". */
export const fmtBits = (b: number): string =>
  `${b.toLocaleString(undefined, { maximumFractionDigits: 1 })} bits`;

/** Format with a fixed number of decimals. */
export const fmt = (x: number, d = 2): string =>
  x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

/** Format a signed delta with explicit sign: "-12.30" / "+4.00". */
export const fmtDelta = (x: number, d = 2): string =>
  `${x >= 0 ? '+' : ''}${fmt(x, d)}`;
