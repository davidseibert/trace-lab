/** Visual helpers for the transformer lens. */

import type { Dataset } from './datasets';

// Match the app theme so activations read as the same blue/orange language as
// the MDL lens's L(M) / L(D|M).
const POS: [number, number, number] = [91, 156, 255]; // --model blue
const NEG: [number, number, number] = [255, 180, 84]; // --data orange
const NEUTRAL = '#8b93a7';

/** Display color for a token, falling back to a neutral grey. */
export function tokenColor(ds: Dataset, token: string): string {
  return ds.tokenColors[token] ?? NEUTRAL;
}

/** Largest absolute value in a buffer — the normaliser for a heatmap. */
export function maxAbs(data: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const x = Math.abs(data[i]);
    if (x > m) m = x;
  }
  return m;
}

/**
 * Cell color for an activation heatmap. Diverging (blue +, orange −) when
 * `signed`, single-hue blue otherwise (e.g. post-ReLU). Alpha encodes magnitude
 * relative to the matrix max, with a small floor so empty cells stay visible.
 */
export function cellColor(v: number, max: number, signed = true): string {
  const a = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  const [r, g, b] = signed ? (v >= 0 ? POS : NEG) : POS;
  return `rgba(${r},${g},${b},${(0.06 + 0.94 * a).toFixed(3)})`;
}
