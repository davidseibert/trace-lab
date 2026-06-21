/**
 * Seeded pseudo-random number generation.
 *
 * The whole point: a transformer's weights are initialised randomly and its
 * training examples are sampled randomly. The reference app used `Math.random()`
 * for both, so every run was different and you could never scrub *backward*
 * through training. By drawing init AND sampling from a single seeded stream,
 * a given (dataset, seed, steps) yields a byte-identical trace — which is what
 * lets us precompute the whole run and make the UI a pure function of the step
 * index, exactly like the MDL lens.
 */

export type Rng = () => number;

/**
 * mulberry32 — a tiny, fast, well-distributed 32-bit PRNG. Deterministic given
 * the seed; returns floats in [0, 1).
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * One draw from a Gaussian N(0, std²) via the Box–Muller transform, using the
 * supplied uniform stream. Used for weight initialisation.
 */
export function gaussian(rng: Rng, std = 1): number {
  let u1 = rng();
  const u2 = rng();
  while (u1 === 0) u1 = rng(); // log(0) guard
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std;
}

/** Uniform integer in [0, n). */
export function randInt(rng: Rng, n: number): number {
  return Math.floor(rng() * n);
}
