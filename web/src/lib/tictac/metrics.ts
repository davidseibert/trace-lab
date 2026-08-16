/**
 * Post-hoc measurement over a finished Tic·tac run — architecture-agnostic:
 * everything reads the TicRun adapter surface (probsAt / unitsAt / sparsity),
 * so the same metrics price a causal GPT, a board encoder, and an MLP without
 * knowing which is which. Three families of readout, all checked against
 * game.ts truth:
 *
 *  - training curves: optimal-move agreement, bits vs the optimal set, and the
 *    D₄ policy-equivariance error, sampled every N steps on a fixed probe suite;
 *  - circuit readouts: unit × ground-truth-feature Pearson correlations and a
 *    2-PC projection (weight sparsity itself lives on TicRun, where the
 *    per-arch parameter layout is known);
 *  - the probe suite itself: seeded, deduped, solver-labeled positions.
 */

import { mulberry32, randInt } from '../llm/rng';
import type { TensorSnap } from '../llm/tensor';
import type { TicRun } from './ticTrain';
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
 * URL reproduces the exact metric numbers. Deliberately includes positions no
 * optimal game visits — agreement measures generalization, not memorization.
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

/** Defense distilled to a predicate: the optimal move is UNIQUE and it
 * completes a block of an opponent two-in-a-row. These are the positions
 * where the arena's defense column was decided (measured: the move-sequence
 * GPT blocks ~69%, board-state archs ~30%). */
export function isForcedBlock(p: ProbePosition): boolean {
  if (p.optimal.length !== 1) return false;
  const opp = toMove(p.board) === 1 ? 2 : 1;
  return LINES.some(
    (l) => l.includes(p.optimal[0]) && l.filter((c) => p.board[c] === opp).length === 2
  );
}

/** A fixed, seeded suite of forced-block positions — the report card's
 * tactical column. Harvested from the same generator as the probe suite
 * (deterministic given the seed), filtered to blocks. */
export function buildBlockSuite(seed: number, n = 24): ProbePosition[] {
  // ~40% of generic probe positions are forced blocks, so 160 → plenty.
  return buildProbeSuite(seed, 160).filter(isForcedBlock).slice(0, n);
}

/**
 * The model's policy at a position: 10-wide token probs from the run, kept on
 * legal cells and renormalized. `illegalMass` is the raw mass on occupied
 * cells and '·' — the "does it know the rules" number (board archs never emit
 * '·', but occupied-cell mass stays meaningful for them).
 */
export function policyAt(
  run: TicRun,
  step: number,
  p: ProbePosition
): { pi: Float64Array; illegalMass: number } {
  const probs = run.probsAt(step, p.moves);
  const pi = new Float64Array(9);
  let legalSum = 0;
  for (const m of p.legal) {
    pi[m] = probs[m + 1];
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
export function equivError(run: TicRun, step: number, p: ProbePosition, g: number): number {
  const base = policyAt(run, step, p).pi;
  const gp: ProbePosition = {
    moves: transformMoves(g, p.moves),
    board: transformBoard(g, p.board),
    legal: transformMoves(g, p.legal),
    optimal: transformMoves(g, p.optimal)
  };
  const trans = policyAt(run, step, gp).pi;
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
  /** Mean raw probability mass on illegal moves (and '·', for the gpt arm). */
  illegalMass: number;
}

export function computeMetrics(run: TicRun, suite: ProbePosition[], every = 10): MetricsPoint[] {
  const out: MetricsPoint[] = [];
  const last = run.steps.length - 1;
  for (let step = 0; step <= last; step += every) {
    out.push(pointAt(run, suite, step));
    if (step !== last && step + every > last) out.push(pointAt(run, suite, last));
  }
  return out;
}

function pointAt(run: TicRun, suite: ProbePosition[], step: number): MetricsPoint {
  let agree = 0;
  let bits = 0;
  let illegal = 0;
  for (const p of suite) {
    const { pi, illegalMass } = policyAt(run, step, p);
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
  for (const p of sub) for (let g = 1; g < 8; g++) equiv += equivError(run, step, p, g);

  return {
    step,
    loss: run.steps[step].loss / Math.LN2,
    agreement: agree / suite.length,
    bitsVsOptimal: bits / suite.length,
    equivariance: equiv / (sub.length * 7),
    sparsity: run.sparsity(step, SPARSITY_THRESHOLD).frac,
    illegalMass: illegal / suite.length
  };
}

// ---------------------------------------------------------------------------
// Circuit readouts
// ---------------------------------------------------------------------------

/** Names for the 26 ground-truth features, aligned with groundTruthFeatures. */
export const FEATURE_NAMES: string[] = [
  ...LINES.map((l) => `X·line ${l.join('')}`),
  ...LINES.map((l) => `O·line ${l.join('')}`),
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

/** Pearson r of each hidden unit's per-position scalar (run.unitsAt) against
 * each ground-truth feature, across the suite → [nUnits, 26] for ActGrid. */
export function unitFeatureCorrelation(run: TicRun, step: number, suite: ProbePosition[]): TensorSnap {
  const n = suite.length;
  const nUnits = run.meta.nUnits;
  const acts: Float64Array[] = [];
  const feats: Float64Array[] = [];
  for (const p of suite) {
    acts.push(run.unitsAt(step, p.moves));
    feats.push(groundTruthFeatures(p));
  }
  const data = new Float64Array(nUnits * 26);
  for (let u = 0; u < nUnits; u++) {
    for (let k = 0; k < 26; k++) {
      data[u * 26 + k] = pearson(
        acts.map((a) => a[u]),
        feats.map((f) => f[k]),
        n
      );
    }
  }
  return { data, shape: [nUnits, 26] };
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

/** Top-2 principal components by power iteration (deterministic start), for
 * the token/cell-embedding scatter. Rows of `m` are points. */
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
    const v = new Float64Array(d);
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
      let norm = 0;
      for (let j = 0; j < d; j++) norm += nv[j] * nv[j];
      norm = Math.sqrt(norm) || 1;
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

/** KL(teacher ‖ student) over legal moves at a position, in bits — the house
 * currency for "how far is the student from the teacher here". */
export function klBits(teacher: TicRun, student: TicRun, p: ProbePosition): number {
  const tq = policyAt(teacher, teacher.steps.length - 1, p).pi;
  const sq = policyAt(student, student.steps.length - 1, p).pi;
  let kl = 0;
  for (const m of p.legal) {
    if (tq[m] > 1e-12) kl += tq[m] * Math.log2(tq[m] / Math.max(sq[m], 1e-12));
  }
  return kl;
}
