/**
 * The arithmetic-coding core, model-agnostic.
 *
 *   ENCODE narrows an interval in [0,1) by a factor of p per symbol. After all
 *   symbols, ANY number inside the final interval names the message; we pick the
 *   shortest binary fraction in it (the "checkout bill" — integer bits ≥ the
 *   ideal Σ-log2(p)).
 *
 *   DECODE zooms back in: at each step it asks which sub-interval the number
 *   falls into, emits that symbol, and rescales. It recovers the symbols purely
 *   from the number + the shared distribution — which is exactly why MDL must
 *   charge L(M): the decoder needs the same model the encoder used.
 *
 * Every step is an immutable snapshot, so the generic `Player` drives playback
 * (including scrubbing backward) with no recomputation.
 */

import { log2 } from '../mdl/format';
import type { CodeDistEntry, CodeStream } from './coder';

export interface CoderStep {
  index: number;
  phase: 'encode' | 'decode';
  /** Interval BEFORE folding this symbol. */
  lo: number;
  hi: number;
  /** Distribution in force at this step. */
  dist: CodeDistEntry[];
  /** Cumulative range of the chosen symbol within [0,1) of the interval. */
  cumLo: number;
  cumHi: number;
  chosenIndex: number;
  chosenLabel: string;
  chosenP: number;
  /** Interval AFTER folding (encode) / the sub-interval located (decode). */
  newLo: number;
  newHi: number;
  /** Ideal bits committed so far = -log2(width after) = Σ surprisal. */
  bitsSoFar: number;
  /** Labels emitted up to and including this step. */
  emitted: string[];
  /** Decode only: the codeword value being read. */
  value?: number;
  /** Decode only: true if the recovered symbol differs from the encoded one. */
  mismatch?: boolean;
  note: string;
}

export interface Codeword {
  /** Binary fraction digits after the point, e.g. "01101" = 0.01101₂. */
  bits: string;
  value: number;
  nbits: number;
}

/** Cumulative-before mass of entry `k` within a distribution. */
function cumulativeBefore(dist: CodeDistEntry[], k: number): number {
  let c = 0;
  for (let j = 0; j < k; j++) c += dist[j].p;
  return c;
}

/** Fewest-bit dyadic fraction k/2^b lying inside [lo, hi). */
export function shortestCodeword(lo: number, hi: number): Codeword {
  for (let b = 1; b <= 53; b++) {
    const scale = 2 ** b;
    const k = Math.ceil(lo * scale);
    const value = k / scale;
    if (value < hi) {
      // Binary digits of k as a b-wide fraction, MSB first. BigInt keeps this
      // exact past 2^31 where JS bitwise ops would wrap.
      let kk = BigInt(k);
      const digits: string[] = [];
      for (let d = 0; d < b; d++) {
        digits.push((kk & 1n).toString());
        kk >>= 1n;
      }
      return { bits: digits.reverse().join(''), value, nbits: b };
    }
  }
  return { bits: '', value: lo, nbits: 53 };
}

/** Encode the stream, returning the per-symbol trace and the final codeword. */
export function encode(stream: CodeStream): { steps: CoderStep[]; codeword: Codeword } {
  let lo = 0;
  let hi = 1;
  const steps: CoderStep[] = [];
  const emitted: string[] = [];

  for (let i = 0; i < stream.length; i++) {
    const sym = stream[i];
    const w = hi - lo;
    const cumLo = cumulativeBefore(sym.dist, sym.chosenIndex);
    const p = sym.dist[sym.chosenIndex].p;
    const cumHi = cumLo + p;
    const newLo = lo + w * cumLo;
    const newHi = lo + w * cumHi;
    emitted.push(sym.label);
    const bitsSoFar = -log2(newHi - newLo);

    steps.push({
      index: i,
      phase: 'encode',
      lo,
      hi,
      dist: sym.dist,
      cumLo,
      cumHi,
      chosenIndex: sym.chosenIndex,
      chosenLabel: sym.label,
      chosenP: p,
      newLo,
      newHi,
      bitsSoFar,
      emitted: [...emitted],
      note: `Fold “${sym.label}” (p=${p.toFixed(3)}, ${(-log2(p)).toFixed(2)} bits) — interval ×${p.toFixed(3)}, ${bitsSoFar.toFixed(2)} bits so far.`
    });

    lo = newLo;
    hi = newHi;
  }

  return { steps, codeword: shortestCodeword(lo, hi) };
}

/**
 * Decode `value` for a message of `stream.length` symbols, using each step's
 * distribution. The emitted symbol is recovered FROM the value (not read off the
 * stream); we compare against the stream only to flag round-trip mismatches.
 */
export function decode(value: number, stream: CodeStream): CoderStep[] {
  let lo = 0;
  let hi = 1;
  const steps: CoderStep[] = [];
  const emitted: string[] = [];

  for (let i = 0; i < stream.length; i++) {
    const dist = stream[i].dist;
    const w = hi - lo;
    const x = (value - lo) / w; // where the codeword sits within this interval

    // Locate which cumulative band contains x.
    let cum = 0;
    let chosenIndex = dist.length - 1;
    for (let j = 0; j < dist.length; j++) {
      if (x < cum + dist[j].p) {
        chosenIndex = j;
        break;
      }
      cum += dist[j].p;
    }
    const cumLo = cumulativeBefore(dist, chosenIndex);
    const p = dist[chosenIndex].p;
    const cumHi = cumLo + p;
    const newLo = lo + w * cumLo;
    const newHi = lo + w * cumHi;
    const label = dist[chosenIndex].label;
    emitted.push(label);
    const bitsSoFar = -log2(newHi - newLo);

    steps.push({
      index: i,
      phase: 'decode',
      lo,
      hi,
      dist,
      cumLo,
      cumHi,
      chosenIndex,
      chosenLabel: label,
      chosenP: p,
      newLo,
      newHi,
      bitsSoFar,
      emitted: [...emitted],
      value,
      mismatch: chosenIndex !== stream[i].chosenIndex,
      note: `Read ${value.toFixed(5)} → lands in “${label}” — emit it, then zoom in.`
    });

    lo = newLo;
    hi = newHi;
  }

  return steps;
}

/** Convenience: total ideal bits of an encode run (final bitsSoFar). */
export function idealBits(steps: CoderStep[]): number {
  return steps.length ? steps[steps.length - 1].bitsSoFar : 0;
}
