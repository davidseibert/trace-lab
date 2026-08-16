/**
 * Post-hoc measurement over a finished Tic·tac training run — everything here
 * replays snapshotted weights through makeForward; trainTrace itself is never
 * touched. Three families of readout, all checked against game.ts truth:
 *
 *  - training curves: optimal-move agreement, bits vs the optimal set, and the
 *    D₄ policy-equivariance error, sampled every N steps on a fixed probe suite;
 *  - circuit readouts: weight-magnitude sparsity (the circuits-repo recipe —
 *    L1 during training, threshold only at inspection), FFN-unit × ground-truth-
 *    feature Pearson correlations, and a 2-PC embedding projection;
 *  - a replay handle (viz + bits ladder at any step, for any board).
 */

import { mulberry32, randInt, type Rng } from '../llm/rng';
import { MiniGPT, loadParams, type ForwardViz } from '../llm/model';
import type { TensorSnap } from '../llm/tensor';
import { logitLens, type LensReport } from '../llm/lens';
import { makeForward, type TrainRun } from '../llm/trainTrace';
import {
  analyze,
  boardFromMoves,
  boardKey,
  generateGame,
  isTerminal,
  legalMoves,
  LINES,
  toMove,
  transformBoard,
  transformMoves,
  TRANSFORMS,
  type Board
} from './game';

export interface ProbePosition {
  moves: number[];
  board: Board;
  legal: number[];
  optimal: number[];
}

/** How many probe positions feed the agreement/bits metrics. */
export const SUITE_SIZE = 40;
/** The first this-many suite positions feed the (8× costlier) equivariance metric. */
export const EQUIV_SUB = 12;
/** |w| below this counts as pruned, both in metrics and the circuit panel default. */
export const SPARSITY_THRESHOLD = 0.01;

/**
 * A fixed, seeded, deduped set of non-terminal positions spread over plies
 * 0..7, harvested from mixed-policy games. Deterministic given the seed, so a
 * URL reproduces the exact metric numbers.
 */
export function buildProbeSuite(seed: number, n: number = SUITE_SIZE): ProbePosition[] {
  const rng = mulberry32(seed + 1000);
  const seen = new Set<number>();
  const out: ProbePosition[] = [];
  let guard = 0;
  while (out.length < n && guard++ < n * 200) {
    const game = generateGame(rng, rng() < 0.5 ? 'optimal' : 'random');
    const cut = randInt(rng, Math.min(game.length, 8)); // plies 0..7
    const moves = game.slice(0, cut);
    const board = boardFromMoves(moves);
    if (isTerminal(board)) continue;
    const key = boardKey(board);
    if (seen.has(key)) continue;
    seen.add(key);
    const a = analyze(board);
    out.push({ moves, board, legal: legalMoves(board), optimal: a.optimal });
  }
  return out;
}

/** Token ids for a move list: '·' (id 0) then cell tokens (cell + 1). */
export const gameIds = (moves: number[]): number[] => [0, ...moves.map((m) => m + 1)];

/**
 * The model's policy at a position: probs at the last input position, kept on
 * legal cells and renormalized. `illegalMass` is the raw mass the model put
 * on occupied cells and '·' — the "does it know the rules" number.
 */
export function policyAt(
  fwd: (step: number, ids: number[]) => ForwardViz,
  step: number,
  p: ProbePosition
): { pi: Float64Array; illegalMass: number } {
  const viz = fwd(step, gameIds(p.moves));
  const pi = new Float64Array(9);
  let legalSum = 0;
  for (const m of p.legal) {
    pi[m] = viz.probs[m + 1];
    legalSum += pi[m];
  }
  if (legalSum > 0) for (const m of p.legal) pi[m] /= legalSum;
  return { pi, illegalMass: Math.max(0, 1 - legalSum) };
}

/**
 * D₄ policy-equivariance error at one (position, transform):
 * TV distance between the pullback g⁻¹·π(g·s) and π(s), both legal-renormalized.
 * 0 = the policy commutes with the board symmetry.
 */
export function equivError(
  fwd: (step: number, ids: number[]) => ForwardViz,
  step: number,
  p: ProbePosition,
  g: number
): number {
  const base = policyAt(fwd, step, p).pi;
  const gp: ProbePosition = {
    moves: transformMoves(g, p.moves),
    board: transformBoard(g, p.board),
    legal: transformMoves(g, p.legal),
    optimal: transformMoves(g, p.optimal)
  };
  const trans = policyAt(fwd, step, gp).pi;
  let tv = 0;
  for (const m of p.legal) tv += Math.abs(trans[TRANSFORMS[g][m]] - base[m]);
  return tv / 2;
}

export interface MetricsPoint {
  step: number;
  /** Mean training loss, in bits (run.steps carries nats). */
  loss: number;
  /** Fraction of the suite where the legal-argmax is a minimax-optimal move. */
  agreement: number;
  /** Mean −log₂ Σ_{m∈optimal} π(m), clamped to 10 bits. */
  bitsVsOptimal: number;
  /** Mean TV pullback error over EQUIV_SUB positions × 7 non-identity g. */
  equivariance: number;
  /** Fraction of all params with |w| < SPARSITY_THRESHOLD. */
  sparsity: number;
  /** Mean raw probability mass on illegal moves and '·'. */
  illegalMass: number;
}

export function computeMetrics(run: TrainRun, suite: ProbePosition[], every = 10): MetricsPoint[] {
  const fwd = makeForward(run);
  const out: MetricsPoint[] = [];
  const last = run.steps.length - 1;
  for (let step = 0; step <= last; step += every) {
    const idx = step; // sample every N plus always the final step below
    out.push(pointAt(run, fwd, suite, idx));
    if (idx !== last && idx + every > last) out.push(pointAt(run, fwd, suite, last));
  }
  return out;
}

function pointAt(
  run: TrainRun,
  fwd: (step: number, ids: number[]) => ForwardViz,
  suite: ProbePosition[],
  step: number
): MetricsPoint {
  let agree = 0;
  let bits = 0;
  let illegal = 0;
  for (const p of suite) {
    const { pi, illegalMass } = policyAt(fwd, step, p);
    illegal += illegalMass;
    let arg = p.legal[0];
    for (const m of p.legal) if (pi[m] > pi[arg]) arg = m;
    if (p.optimal.includes(arg)) agree++;
    let optMass = 0;
    for (const m of p.optimal) optMass += pi[m];
    bits += Math.min(10, -Math.log2(Math.max(optMass, 2 ** -10)));
  }
  let equiv = 0;
  const sub = suite.slice(0, EQUIV_SUB);
  for (const p of sub) for (let g = 1; g < 8; g++) equiv += equivError(fwd, step, p, g);

  const w = run.weights[step];
  let small = 0;
  for (let i = 0; i < w.length; i++) if (Math.abs(w[i]) < SPARSITY_THRESHOLD) small++;

  return {
    step,
    loss: run.steps[step].loss / Math.LN2,
    agreement: agree / suite.length,
    bitsVsOptimal: bits / suite.length,
    equivariance: equiv / (sub.length * 7),
    sparsity: small / w.length,
    illegalMass: illegal / suite.length
  };
}

// ---------------------------------------------------------------------------
// Circuit readouts
// ---------------------------------------------------------------------------

/** Names for the 26 ground-truth features, aligned with groundTruthFeatures. */
export const FEATURE_NAMES: string[] = [
  ...LINES.map((l, i) => `X·line ${l.join('')}`),
  ...LINES.map((l, i) => `O·line ${l.join('')}`),
  'to move',
  ...[...Array(9).keys()].map((c) => `cell ${c}`)
];

/**
 * 26 interpretable features of a position: per line × player, the player's
 * mark count on that line if the opponent is absent from it (else 0 — a
 * blocked line threatens nothing); player-to-move ±1; and the 9 cell
 * occupancies (+1 X, −1 O, 0 empty). The known feature basis of the game —
 * what a "line detector" unit would have to correlate with.
 */
export function groundTruthFeatures(p: ProbePosition): Float64Array {
  const f = new Float64Array(26);
  LINES.forEach((line, i) => {
    let x = 0;
    let o = 0;
    for (const c of line) {
      if (p.board[c] === 1) x++;
      else if (p.board[c] === 2) o++;
    }
    f[i] = o === 0 ? x : 0;
    f[8 + i] = x === 0 ? o : 0;
  });
  f[16] = toMove(p.board) === 1 ? 1 : -1;
  for (let c = 0; c < 9; c++) f[17 + c] = p.board[c] === 1 ? 1 : p.board[c] === 2 ? -1 : 0;
  return f;
}

/** Pearson r of each FFN hidden unit's last-position activation against each
 * ground-truth feature, across the suite → [ffnHid, 26] for ActGrid. */
export function unitFeatureCorrelation(
  fwd: (step: number, ids: number[]) => ForwardViz,
  step: number,
  suite: ProbePosition[],
  ffnHid: number
): TensorSnap {
  const n = suite.length;
  const acts: Float64Array[] = []; // per position: [ffnHid]
  const feats: Float64Array[] = [];
  for (const p of suite) {
    const viz = fwd(step, gameIds(p.moves));
    const T = viz.ffnHidden.shape[0];
    const row = new Float64Array(ffnHid);
    for (let u = 0; u < ffnHid; u++) row[u] = viz.ffnHidden.data[(T - 1) * ffnHid + u];
    acts.push(row);
    feats.push(groundTruthFeatures(p));
  }
  const data = new Float64Array(ffnHid * 26);
  for (let u = 0; u < ffnHid; u++) {
    for (let k = 0; k < 26; k++) {
      data[u * 26 + k] = pearson(
        acts.map((a) => a[u]),
        feats.map((f) => f[k]),
        n
      );
    }
  }
  return { data, shape: [ffnHid, 26] };
}

function pearson(xs: number[], ys: number[], n: number): number {
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const den = Math.sqrt(sxx * syy);
  return den < 1e-12 ? 0 : sxy / den;
}

export interface SparsityReport {
  /** Fraction of all params below the threshold. */
  frac: number;
  /** FFN hidden units with at least one surviving inbound AND outbound edge. */
  liveUnits: number;
  /** Total surviving FFN edges (fc1 + fc2). */
  edges: number;
}

/**
 * The circuits-repo readout: threshold |w|, count what survives. Works on the
 * flat snapshot; fc1/fc2 offsets are located from the run's config using the
 * flattenParams ordering (tokEmb, posEmb, block[ln1, attn(wq,wk,wv,wo ×(w,b)),
 * ln2, ffn(fc1 w,b, fc2 w,b)], lnFinal, head) — rather than hardcoding offsets
 * we scan for the two FFN-shaped segments by size, which is stable as long as
 * no other param shares those exact sizes (true for this config).
 */
export function sparsityReport(
  run: TrainRun,
  weights: Float64Array,
  threshold: number
): SparsityReport {
  let small = 0;
  for (let i = 0; i < weights.length; i++) if (Math.abs(weights[i]) < threshold) small++;

  // Locate fc1 [D, F] and fc2 [F, D] inside the flat vector by replaying
  // param order on a throwaway model — exact, no size-scanning ambiguity.
  const model = new MiniGPT(run.cfg, mulberry32(1));
  const params = model.params();
  const D = run.cfg.embDim;
  const F = run.cfg.ffnHid;
  let off = 0;
  let fc1: { off: number } | null = null;
  let fc2: { off: number } | null = null;
  for (const p of params) {
    const [r, c] = p.shape.length === 2 ? p.shape : [1, p.shape[0]];
    if (r === D && c === F && !fc1) fc1 = { off };
    else if (r === F && c === D && !fc2) fc2 = { off };
    off += p.data.length;
  }
  if (!fc1 || !fc2) return { frac: small / weights.length, liveUnits: 0, edges: 0 };

  let edges = 0;
  let live = 0;
  for (let u = 0; u < F; u++) {
    let inn = 0;
    let out = 0;
    for (let d = 0; d < D; d++) {
      if (Math.abs(weights[fc1.off + d * F + u]) >= threshold) inn++;
      if (Math.abs(weights[fc2.off + u * D + d]) >= threshold) out++;
    }
    edges += inn + out;
    if (inn > 0 && out > 0) live++;
  }
  return { frac: small / weights.length, liveUnits: live, edges };
}

/** Top-2 principal components by power iteration (deterministic start), for
 * the token-embedding scatter. Rows of `m` are points. */
export function pca2(m: TensorSnap): { x: number; y: number }[] {
  const [n, d] = m.shape;
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += m.data[i * d + j] / n;
  const X = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = m.data[i * d + j] - mean[j];

  const project = (v: Float64Array) => {
    const s = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) s[i] += X[i * d + j] * v[j];
    return s;
  };
  const powerIter = (deflate: Float64Array | null): Float64Array => {
    let v = new Float64Array(d);
    for (let j = 0; j < d; j++) v[j] = Math.sin(j + 1); // deterministic start
    for (let it = 0; it < 60; it++) {
      if (deflate) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += v[j] * deflate[j];
        for (let j = 0; j < d; j++) v[j] -= dot * deflate[j];
      }
      const s = project(v);
      const nv = new Float64Array(d);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) nv[j] += X[i * d + j] * s[i];
      let norm = Math.hypot(...nv);
      if (norm < 1e-12) norm = 1;
      for (let j = 0; j < d; j++) v[j] = nv[j] / norm;
    }
    return v;
  };
  const p1 = powerIter(null);
  const p2 = powerIter(p1);
  const s1 = project(p1);
  const s2 = project(p2);
  return [...Array(n).keys()].map((i) => ({ x: s1[i], y: s2[i] }));
}

// ---------------------------------------------------------------------------
// Replay handle
// ---------------------------------------------------------------------------

export interface Replay {
  viz(step: number, ids: number[]): ForwardViz;
  lens(step: number, ids: number[]): LensReport;
  /** The full token-embedding table [vocab, embDim] at a step — the PCA input. */
  tokenTable(step: number): TensorSnap;
}

/** One reused model instance; every panel's on-demand "what did the net do at
 * step i on THIS board" goes through here. Simpler than makeLab — no facts. */
export function makeReplay(run: TrainRun): Replay {
  const model = new MiniGPT(run.cfg, mulberry32(1));
  const viz = (step: number, ids: number[]): ForwardViz => {
    loadParams(model, run.weights[step]);
    model.forward(ids);
    return model.viz!;
  };
  return {
    viz,
    lens(step, ids) {
      viz(step, ids);
      return logitLens(model);
    },
    tokenTable(step) {
      loadParams(model, run.weights[step]);
      return model.tokenEmb.weight.snapshot();
    }
  };
}
