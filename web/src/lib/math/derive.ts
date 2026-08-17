import type { CodeStream } from '../coder/coder';
import { surprisal } from '../mdl/format';

const positive = (p: number): number => Math.max(p, Number.MIN_VALUE);

/** Information in one observed event. A zero-probability observation is impossible. */
export const informationBits = (p: number): number =>
  surprisal(p);

export function normalize(weights: readonly number[]): number[] {
  const clean = weights.map((w) => Math.max(0, Number.isFinite(w) ? w : 0));
  if (clean.length === 0) return [];
  const total = clean.reduce((a, b) => a + b, 0);
  return total > 0 ? clean.map((w) => w / total) : clean.map(() => 1 / clean.length);
}

export function entropyBits(p: readonly number[]): number {
  return p.reduce((h, q) => h + (q > 0 ? q * informationBits(q) : 0), 0);
}

export function crossEntropyBits(q: readonly number[], p: readonly number[]): number {
  return q.reduce((h, qi, i) => h + (qi > 0 ? qi * informationBits(positive(p[i] ?? 0)) : 0), 0);
}

export function klDivergenceBits(q: readonly number[], p: readonly number[]): number {
  return crossEntropyBits(q, p) - entropyBits(q);
}

export const effectiveChoices = (entropy: number): number => 2 ** entropy;

/** A two-symbol memoryless source for the foundations arithmetic-coding lab. */
export function binaryStream(message: string, pA: number): CodeStream {
  const a = Math.min(0.99, Math.max(0.01, pA));
  const dist = [
    { id: 0, label: 'A', p: a },
    { id: 1, label: 'B', p: 1 - a }
  ];
  return [...message].map((label) => ({
    label,
    dist,
    chosenIndex: label === 'A' ? 0 : 1
  }));
}
