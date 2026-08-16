/**
 * Modern continuous Hopfield networks (Ramsauer et al. 2020, "Hopfield
 * Networks is All You Need") plus the classical binary network as the
 * historical baseline — pure math, no Svelte imports, `TensorSnap` only at
 * the UI boundary.
 *
 * The modern update over stored patterns X (rows) and state ξ is
 *
 *     ξ_new = Xᵀ · softmax(β · X ξ)
 *
 * which is literally one row of transformer attention with the patterns as
 * both keys and values and β playing 1/√d_k. Its energy
 *
 *     E(ξ) = −(1/β)·lse(β, Xξ) + ½‖ξ‖² + (1/β)·ln N + ½M²,   M = max‖xᵢ‖
 *
 * provably decreases under the update (CCCP), and with the constants kept —
 * most presentations drop them — E ≥ 0, so it charts on the house [0, max]
 * scale and converts to bits by ÷ln 2.
 *
 * Everything random funnels through a caller-supplied `Rng`, so a given
 * (patterns, seed, knobs) tuple yields a byte-identical trace — the UI stays
 * a pure function of the step index, exactly like every other lens.
 */

import type { TensorSnap } from '../llm/tensor';
import { softmaxVec } from '../llm/tensor';
import type { Rng } from '../llm/rng';
import { gaussian, mulberry32, randInt } from '../llm/rng';

export type Regime = 'retrieval' | 'metastable' | 'global';

/** effK = 2^H(weights): ≤ this ⇒ one pattern dominates ⇒ 'retrieval'. */
export const RETRIEVAL_EFFK = 1.5;
/** effK ≥ this fraction of N ⇒ near-uniform mixing ⇒ 'global'. */
export const GLOBAL_EFFK_FRAC = 0.75;
/** Modern retrieval counts as a success at this cosine to the intended pattern. */
export const MODERN_SUCCESS_COS = 0.9;
/** Classical retrieval counts as a success at this sign-agreement fraction. */
export const CLASSICAL_SUCCESS_AGREE = 0.95;

export interface ModernStep {
  /** 0 = the initial (corrupted) state, before any update. */
  iter: number;
  xi: Float64Array;
  /** Xξ — one raw dot product per stored pattern (the "scores" row). */
  overlaps: Float64Array;
  /** softmax(β·overlaps) — the attention row over patterns. */
  weights: Float64Array;
  energyNats: number;
  /** energyNats / ln 2 — the house currency. */
  energyBits: number;
  /** H(weights) in bits; 2^H = effective number of mixed patterns. */
  entropyBits: number;
  maxWeight: number;
  /** ‖ξ − ξ_prev‖₂; 0 at iter 0. */
  deltaNorm: number;
}

export interface ClassicalStep {
  /** 0 = initial state; each later step is one full asynchronous sweep. */
  iter: number;
  /** ±1 state after the sweep. */
  xi: Float64Array;
  /** (1/d)·Xξ — per-pattern normalized overlap mᵢ ∈ [−1, 1]. */
  overlaps: Float64Array;
  /** −½ ξᵀWξ. Unitless; charted on its own normalized scale. */
  energy: number;
  /** Sign flips during the sweep; 0 ⇒ converged. */
  flips: number;
}

export interface HopfieldTrace<S> {
  steps: S[];
  converged: boolean;
  /** Best-matching stored pattern at the final step, if it clears the
   * success threshold (cosine for modern, sign agreement for classical). */
  retrieved: number | null;
  /** Regime of the final step's weights (modern); classical reports it from
   * normalized overlaps pushed through the same classifier for comparability. */
  regime: Regime;
}

/** One playback step for the zipped modern/classical player. */
export interface ComboStep {
  iter: number;
  modern: ModernStep | null;
  classical: ClassicalStep | null;
}

const LN2 = Math.LN2;

// ---------------------------------------------------------------------------
// Small vector helpers (patterns are rows of X: TensorSnap [N, d])
// ---------------------------------------------------------------------------

function rowDot(X: TensorSnap, row: number, v: Float64Array): number {
  const d = X.shape[1];
  let s = 0;
  for (let j = 0; j < d; j++) s += X.data[row * d + j] * v[j];
  return s;
}

/** Xξ — one dot product per stored pattern. */
export function overlapsOf(X: TensorSnap, xi: Float64Array): Float64Array {
  const N = X.shape[0];
  const o = new Float64Array(N);
  for (let i = 0; i < N; i++) o[i] = rowDot(X, i, xi);
  return o;
}

function norm2(v: Float64Array): number {
  let s = 0;
  for (let j = 0; j < v.length; j++) s += v[j] * v[j];
  return Math.sqrt(s);
}

function cosine(a: Float64Array, b: Float64Array): number {
  let dot = 0;
  for (let j = 0; j < a.length; j++) dot += a[j] * b[j];
  const den = norm2(a) * norm2(b);
  return den === 0 ? 0 : dot / den;
}

// ---------------------------------------------------------------------------
// Modern (continuous) network
// ---------------------------------------------------------------------------

export function modernUpdate(
  X: TensorSnap,
  xi: Float64Array,
  beta: number
): { xiNext: Float64Array; overlaps: Float64Array; weights: Float64Array } {
  const [N, d] = X.shape;
  const overlaps = overlapsOf(X, xi);
  const scaled = new Float64Array(N);
  for (let i = 0; i < N; i++) scaled[i] = beta * overlaps[i];
  const weights = softmaxVec(scaled);
  // ξ_next = Xᵀw — a convex mix of the stored patterns.
  const xiNext = new Float64Array(d);
  for (let i = 0; i < N; i++) {
    const w = weights[i];
    for (let j = 0; j < d; j++) xiNext[j] += w * X.data[i * d + j];
  }
  return { xiNext, overlaps, weights };
}

/**
 * The paper's energy with its constants INCLUDED — (1/β)·ln N + ½M² — so
 * E ≥ 0 (Ramsauer et al., Lemma on energy bounds) and the chart's [0, max]
 * mapping needs no shifting. lse is computed stably (subtract the max).
 */
export function modernEnergy(X: TensorSnap, xi: Float64Array, beta: number): number {
  const [N, d] = X.shape;
  const o = overlapsOf(X, xi);
  let mx = -Infinity;
  for (let i = 0; i < N; i++) mx = Math.max(mx, beta * o[i]);
  let s = 0;
  for (let i = 0; i < N; i++) s += Math.exp(beta * o[i] - mx);
  const lse = (mx + Math.log(s)) / beta;
  let M = 0;
  for (let i = 0; i < N; i++) {
    let r = 0;
    for (let j = 0; j < d; j++) r += X.data[i * d + j] ** 2;
    M = Math.max(M, r);
  }
  const xiSq = xi.reduce((acc, v) => acc + v * v, 0);
  return -lse + 0.5 * xiSq + Math.log(N) / beta + 0.5 * M;
}

/** Shannon entropy of a weight vector, in bits. */
export function entropyBits(weights: Float64Array): number {
  let h = 0;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (w > 0) h -= w * Math.log2(w);
  }
  return h;
}

/**
 * Retrieval regimes by effective pattern count 2^H: one dominant pattern
 * (retrieval), a near-uniform blend of everything (global fixed point), or
 * the interesting middle — a metastable average over a subset.
 */
export function classifyRegime(weights: Float64Array, N: number): Regime {
  const effK = 2 ** entropyBits(weights);
  if (effK <= RETRIEVAL_EFFK) return 'retrieval';
  if (effK >= GLOBAL_EFFK_FRAC * N) return 'global';
  return 'metastable';
}

export function runModern(
  X: TensorSnap,
  xi0: Float64Array,
  beta: number,
  opts: { maxIters?: number; tol?: number } = {}
): HopfieldTrace<ModernStep> {
  // 24 iters is generous — clean retrieval is ε-done after one — but lets a
  // metastable blend actually settle instead of reporting "not converged".
  const { maxIters = 24, tol = 1e-5 } = opts;
  const N = X.shape[0];

  const mkStep = (iter: number, xi: Float64Array, deltaNorm: number): ModernStep => {
    const overlaps = overlapsOf(X, xi);
    const scaled = new Float64Array(N);
    for (let i = 0; i < N; i++) scaled[i] = beta * overlaps[i];
    const weights = softmaxVec(scaled);
    const energyNats = modernEnergy(X, xi, beta);
    let maxWeight = 0;
    for (let i = 0; i < N; i++) maxWeight = Math.max(maxWeight, weights[i]);
    return {
      iter,
      xi,
      overlaps,
      weights,
      energyNats,
      energyBits: energyNats / LN2,
      entropyBits: entropyBits(weights),
      maxWeight,
      deltaNorm
    };
  };

  // Iter 0 records the initial state so the chart shows the one-step cliff.
  const steps: ModernStep[] = [mkStep(0, xi0, 0)];
  let converged = false;
  let xi = xi0;
  for (let t = 1; t <= maxIters; t++) {
    const { xiNext } = modernUpdate(X, xi, beta);
    let dn = 0;
    for (let j = 0; j < xi.length; j++) dn += (xiNext[j] - xi[j]) ** 2;
    dn = Math.sqrt(dn);
    xi = xiNext;
    steps.push(mkStep(t, xi, dn));
    if (dn < tol) {
      converged = true;
      break;
    }
  }

  const last = steps[steps.length - 1];
  const retrieved = bestMatch(X, last.xi, (i) => cosine(last.xi, patternRow(X, i)) >= MODERN_SUCCESS_COS);
  return { steps, converged, retrieved, regime: classifyRegime(last.weights, N) };
}

function patternRow(X: TensorSnap, i: number): Float64Array {
  const d = X.shape[1];
  return X.data.subarray(i * d, (i + 1) * d) as Float64Array;
}

function bestMatch(X: TensorSnap, xi: Float64Array, accept: (i: number) => boolean): number | null {
  const N = X.shape[0];
  let best = -1;
  let bestCos = -Infinity;
  for (let i = 0; i < N; i++) {
    const c = cosine(xi, patternRow(X, i));
    if (c > bestCos) {
      bestCos = c;
      best = i;
    }
  }
  return best >= 0 && accept(best) ? best : null;
}

// ---------------------------------------------------------------------------
// Classical (binary) network
// ---------------------------------------------------------------------------

/** Hebbian weights W = (1/d)·Σ xxᵀ with zero diagonal. Callers pass ±1 rows
 * (sign() any continuous patterns first — never inside the modern path). */
export function hebbianWeights(X: TensorSnap): Float64Array {
  const [N, d] = X.shape;
  const W = new Float64Array(d * d);
  for (let i = 0; i < N; i++)
    for (let a = 0; a < d; a++) {
      const xa = X.data[i * d + a];
      for (let b = 0; b < d; b++) W[a * d + b] += (xa * X.data[i * d + b]) / d;
    }
  for (let a = 0; a < d; a++) W[a * d + a] = 0;
  return W;
}

/**
 * Asynchronous sequential sweeps in fixed index order — deterministic, energy
 * monotonically non-increasing, and immune to the synchronous update's
 * 2-cycles. sign(0) keeps the current sign (also deterministic). Takes X (not
 * W) so each step can report the per-pattern overlaps the UI shows.
 */
export function runClassical(
  X: TensorSnap,
  xi0: Float64Array,
  opts: { maxSweeps?: number } = {}
): HopfieldTrace<ClassicalStep> {
  const { maxSweeps = 32 } = opts;
  const [N, d] = X.shape;
  const W = hebbianWeights(X);

  const mkStep = (iter: number, xi: Float64Array, flips: number): ClassicalStep => {
    const overlaps = overlapsOf(X, xi);
    for (let i = 0; i < N; i++) overlaps[i] /= d;
    let e = 0;
    for (let a = 0; a < d; a++) {
      let field = 0;
      for (let b = 0; b < d; b++) field += W[a * d + b] * xi[b];
      e += xi[a] * field;
    }
    return { iter, xi, overlaps, energy: -0.5 * e, flips };
  };

  const steps: ClassicalStep[] = [mkStep(0, xi0, 0)];
  let converged = false;
  const xi = Float64Array.from(xi0);
  for (let t = 1; t <= maxSweeps; t++) {
    let flips = 0;
    for (let a = 0; a < d; a++) {
      let field = 0;
      for (let b = 0; b < d; b++) field += W[a * d + b] * xi[b];
      const next = field > 0 ? 1 : field < 0 ? -1 : xi[a];
      if (next !== xi[a]) {
        xi[a] = next;
        flips++;
      }
    }
    steps.push(mkStep(t, Float64Array.from(xi), flips));
    if (flips === 0) {
      converged = true;
      break;
    }
  }

  const last = steps[steps.length - 1];
  const retrieved = bestMatch(X, last.xi, (i) => signAgreement(last.xi, patternRow(X, i)) >= CLASSICAL_SUCCESS_AGREE);
  // Same classifier as modern, fed |overlap|-softmax-free proportions: use
  // normalized positive overlaps as pseudo-weights purely for the regime chip.
  const pos = Float64Array.from(last.overlaps, (m) => Math.max(m, 0));
  const sum = pos.reduce((a, v) => a + v, 0) || 1;
  for (let i = 0; i < N; i++) pos[i] /= sum;
  return { steps, converged, retrieved, regime: classifyRegime(pos, N) };
}

function signAgreement(a: Float64Array, b: Float64Array): number {
  let agree = 0;
  for (let j = 0; j < a.length; j++) if (Math.sign(a[j]) === Math.sign(b[j])) agree++;
  return agree / a.length;
}

/** Zip modern and classical traces for one player, padding the shorter one by
 * holding its converged final step. */
export function zipTraces(
  modern: HopfieldTrace<ModernStep> | null,
  classical: HopfieldTrace<ClassicalStep> | null
): ComboStep[] {
  const len = Math.max(modern?.steps.length ?? 0, classical?.steps.length ?? 0);
  const out: ComboStep[] = [];
  for (let t = 0; t < len; t++)
    out.push({
      iter: t,
      modern: modern ? modern.steps[Math.min(t, modern.steps.length - 1)] : null,
      classical: classical ? classical.steps[Math.min(t, classical.steps.length - 1)] : null
    });
  return out;
}

// ---------------------------------------------------------------------------
// Patterns, corruption, separation
// ---------------------------------------------------------------------------

/** Random ±1 patterns — the same rows serve both networks, and their norm √d
 * makes β* = 1/√d the natural scale for the modern overlaps. */
export function randomPatterns(n: number, d: number, rng: Rng): TensorSnap {
  const data = new Float64Array(n * d);
  for (let i = 0; i < data.length; i++) data[i] = gaussian(rng) >= 0 ? 1 : -1;
  return { data, shape: [n, d] };
}

/** Δ = min_i (xᵢᵀxᵢ − max_{j≠i} xᵢᵀxⱼ) — the paper's separation quantity;
 * capacity theorems charge for Δ, not for raw N. */
export function separationStats(X: TensorSnap): { delta: number; minPair: [number, number] } {
  const [N, d] = X.shape;
  let delta = Infinity;
  let minPair: [number, number] = [0, 0];
  for (let i = 0; i < N; i++) {
    const self = rowDot(X, i, patternRow(X, i));
    let worst = -Infinity;
    let worstJ = i;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const o = rowDot(X, i, patternRow(X, j));
      if (o > worst) {
        worst = o;
        worstJ = j;
      }
    }
    const sep = N > 1 ? self - worst : self;
    if (sep < delta) {
      delta = sep;
      minPair = [i, worstJ];
    }
  }
  return { delta, minPair };
}

/**
 * Corrupt a stored ±1 pattern into the two initial states the two networks
 * need: the same seeded sign flips for both; a masked bottom half becomes 0
 * for the modern state ("no information here") but random ±1 for the
 * classical one — 0 is not a valid binary state.
 */
export function corrupt(
  xi: Float64Array,
  rng: Rng,
  opts: { flipFrac: number; mask?: 'none' | 'bottom' }
): { modern: Float64Array; classical: Float64Array } {
  const d = xi.length;
  const modern = Float64Array.from(xi);
  const flips = Math.round(opts.flipFrac * d);
  const idx = [...Array(d).keys()];
  // Partial Fisher–Yates: the first `flips` entries are a uniform sample.
  for (let k = 0; k < flips; k++) {
    const r = k + randInt(rng, d - k);
    [idx[k], idx[r]] = [idx[r], idx[k]];
    modern[idx[k]] = -modern[idx[k]];
  }
  const classical = Float64Array.from(modern);
  if (opts.mask === 'bottom') {
    for (let j = Math.floor(d / 2); j < d; j++) {
      modern[j] = 0;
      classical[j] = rng() < 0.5 ? -1 : 1;
    }
  }
  return { modern, classical };
}

// ---------------------------------------------------------------------------
// β sweep and capacity experiment
// ---------------------------------------------------------------------------

export interface BetaSweepPoint {
  beta: number;
  entropyBits: number;
  maxWeight: number;
  regime: Regime;
}

/** Log-spaced β grid centered on β* = 1/√d — the value attention uses. */
export function logSpacedBetas(betaStar: number, decadesDown = 2, decadesUp = 2, count = 33): number[] {
  const lo = Math.log10(betaStar) - decadesDown;
  const span = decadesDown + decadesUp;
  return [...Array(count).keys()].map((i) => 10 ** (lo + (span * i) / (count - 1)));
}

/** Final-step statistics of a full retrieval run at each β. */
export function betaSweep(X: TensorSnap, xi0: Float64Array, betas: number[]): BetaSweepPoint[] {
  return betas.map((beta) => {
    const t = runModern(X, xi0, beta);
    const last = t.steps[t.steps.length - 1];
    return { beta, entropyBits: last.entropyBits, maxWeight: last.maxWeight, regime: t.regime };
  });
}

export interface CapacityPoint {
  N: number;
  modern: number;
  classical: number;
}

/**
 * Success rate vs stored-pattern count, modern and classical on identical
 * patterns and corruptions. Each (N, trial) cell reseeds independently, so
 * any single cell is reproducible on its own.
 */
export function capacityExperiment(opts: {
  Ns: number[];
  trials: number;
  noiseFrac: number;
  beta: number;
  seed: number;
  /** Draw patterns for one trial: N seeded ±1 rows. */
  draw: (N: number, rng: Rng) => TensorSnap;
}): CapacityPoint[] {
  const { Ns, trials, noiseFrac, beta, seed, draw } = opts;
  return Ns.map((N) => {
    let okModern = 0;
    let okClassical = 0;
    for (let t = 0; t < trials; t++) {
      const rng = mulberryCell(seed, N, t);
      const X = draw(N, rng);
      const target = randInt(rng, N);
      const { modern, classical } = corrupt(patternRow(X, target), rng, { flipFrac: noiseFrac });
      if (runModern(X, modern, beta).retrieved === target) okModern++;
      if (runClassical(X, classical).retrieved === target) okClassical++;
    }
    return { N, modern: okModern / trials, classical: okClassical / trials };
  });
}

// Each capacity cell reseeds independently of every other, so a single (N,
// trial) point can be reproduced without replaying the whole sweep.
function mulberryCell(seed: number, N: number, trial: number): Rng {
  return mulberry32(seed * 1e4 + N * 100 + trial);
}
