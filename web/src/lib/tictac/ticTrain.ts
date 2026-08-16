/**
 * One trainer for the whole Tic·tac arch × signal grid. Every cell emits the
 * same `TicRun` shape, whose adapter surface (probsAt / unitsAt / vizAt /
 * sparsity) is what the metrics module and every panel consume — so the same
 * charts read a causal GPT over move sequences, a bidirectional board
 * encoder, and a plain MLP without knowing which is which.
 *
 * Signals:
 *  - games  — hard targets sampled from the corpus (the original setup).
 *  - solver — SOFT targets: uniform over the minimax-optimal moves. The same
 *    objective as `games` in expectation, with the sampling noise removed —
 *    the noiseless estimator (this is what should cure the opening wobble).
 *  - distill — soft targets from a trained gpt+games teacher, tempered by
 *    q^(1/T) renormalization (softmax temperature without logits — the same
 *    identity Hopfield·heads uses).
 *
 * Determinism: one seeded rng for init + example sampling, a separate one for
 * the eval subsample (mirroring trainTrace's convention exactly, so the
 * gpt+games cell reproduces trainTrace's loss curve bit-for-bit).
 */

import { mulberry32, randInt, type Rng } from '../llm/rng';
import { MiniGPT, Adam, type ForwardViz, type ModelConfig } from '../llm/model';
import {
  Tensor,
  backward,
  crossEntropyLoss,
  softCrossEntropyLoss,
  type TensorSnap
} from '../llm/tensor';
import { logitLens, type LensReport } from '../llm/lens';
import { analyze, boardFromMoves, boardKey, generateCorpus, type Board, type CorpusKind } from './game';
import { makeBoardModel, MLP_HID, ENC_FFN, type BoardArch, type BoardModel, type BoardViz } from './boardModels';
import { TIC_VOCAB } from './dataset';

export type Arch = 'gpt' | BoardArch;
export type Signal = 'games' | 'solver' | 'distill';

export interface TicStep {
  index: number;
  /** Mean eval-subsample loss under this cell's own objective, in nats. */
  loss: number;
}

export interface SparsityReport {
  frac: number;
  liveUnits: number;
  edges: number;
}

export interface TicRun {
  arch: Arch;
  signal: Signal;
  steps: TicStep[];
  weights: Float64Array[];
  /** For the gpt arm, the MiniGPT config (panels read nHeads etc.). */
  cfg: ModelConfig | null;
  meta: { nHeads: number; nUnits: number; paramCount: number };
  /** Raw next-token probs, ALWAYS 10-wide ('·' + 9 cells); board archs put 0 on '·'. */
  probsAt(step: number, moves: number[]): Float64Array;
  /** One scalar per hidden unit for this position (gpt: last position's FFN row;
   * encoder: mean over the 9 cells; mlp: the single row). */
  unitsAt(step: number, moves: number[]): Float64Array;
  vizAt(step: number, moves: number[]): ForwardViz | BoardViz;
  /** gpt only — the LensView ladder; null for board archs. */
  lensAt(step: number, moves: number[]): LensReport | null;
  /** PCA input: gpt token table [10,d] · encoder cellEmb [9,d] · mlp per-cell
   * X-state input weight rows [9,hid]. */
  tokenTable(step: number): TensorSnap;
  sparsity(step: number, thr: number): SparsityReport;
}

export interface TicTrainOptions {
  arch: Arch;
  signal: Signal;
  kind: CorpusKind;
  nGames: number;
  steps: number;
  seed: number;
  l1Lambda: number;
  /** Distillation temperature (q^(1/T) renorm on the teacher's policy). */
  temperature?: number;
  teacher?: TicRun;
}

const GPT_CFG: ModelConfig = { vocabSize: 10, embDim: 24, ctxLen: 10, nHeads: 2, ffnHid: 48 };
const LR = 0.01;
const EVAL_SAMPLE = 50;

const flatten = (params: Tensor[]): Float64Array => {
  const n = params.reduce((s, p) => s + p.data.length, 0);
  const out = new Float64Array(n);
  let off = 0;
  for (const p of params) {
    out.set(p.data, off);
    off += p.data.length;
  }
  return out;
};

const load = (params: Tensor[], flat: Float64Array): void => {
  let off = 0;
  for (const p of params) {
    p.data.set(flat.subarray(off, off + p.data.length));
    off += p.data.length;
  }
};

export const gameIds = (moves: number[]): number[] => [0, ...moves.map((m) => m + 1)];

/** Uniform-over-optimal distribution for a board, over 9 cells. */
function solverQ9(b: Board): Float64Array {
  const q = new Float64Array(9);
  const opt = analyze(b).optimal;
  for (const m of opt) q[m] = 1 / opt.length;
  return q;
}

/** Teacher policy over 9 cells, tempered: (p^(1/T)) / Σ p^(1/T). */
function teacherQ9(teacher: TicRun, moves: number[], T: number): Float64Array {
  const p10 = teacher.probsAt(teacher.steps.length - 1, moves);
  const q = new Float64Array(9);
  let s = 0;
  for (let m = 0; m < 9; m++) {
    q[m] = Math.max(p10[m + 1], 1e-12) ** (1 / T);
    s += q[m];
  }
  for (let m = 0; m < 9; m++) q[m] /= s;
  return q;
}

export function ticTrain(opts: TicTrainOptions): TicRun {
  const { arch, signal, kind, nGames, steps, seed, l1Lambda } = opts;
  const T = opts.temperature ?? 2;
  if (signal === 'distill' && (!opts.teacher || arch === 'gpt'))
    throw new Error('distill needs a teacher and a board-model student');

  const corpus = generateCorpus(kind, nGames, seed).map((seq) => seq.map((t) => TIC_VOCAB.indexOf(t)));
  const rng: Rng = mulberry32(seed);

  return arch === 'gpt'
    ? trainGpt(corpus, signal, rng, seed, steps, l1Lambda)
    : trainBoard(arch, corpus, signal, rng, seed, steps, l1Lambda, opts.teacher ?? null, T);
}

/** Pick a fixed eval subsample from its own rng stream (trainTrace convention:
 * setting it never shifts init or example sampling). */
function evalSubset<I>(items: I[], seed: number): I[] {
  if (items.length <= EVAL_SAMPLE) return items;
  const sub = mulberry32(seed ^ 0x9e3779b9);
  const idx = [...items.keys()];
  for (let k = 0; k < EVAL_SAMPLE; k++) {
    const r = k + randInt(sub, idx.length - k);
    [idx[k], idx[r]] = [idx[r], idx[k]];
  }
  return idx.slice(0, EVAL_SAMPLE).map((i) => items[i]);
}

function l1Inject(params: Tensor[], lambda: number): void {
  if (!lambda) return;
  for (const p of params) for (let j = 0; j < p.data.length; j++) p.grad[j] += lambda * Math.sign(p.data[j]);
}

// ---------------------------------------------------------------------------
// GPT arm — sequences (ported from trainTrace, minus the Dataset machinery)
// ---------------------------------------------------------------------------

function trainGpt(
  corpus: number[][],
  signal: Signal,
  rng: Rng,
  seed: number,
  steps: number,
  l1Lambda: number
): TicRun {
  const model = new MiniGPT(GPT_CFG, rng);
  const params = model.params();
  const optimizer = new Adam(params, LR);

  // Pre-shift, and for solver-soft precompute each position's target row:
  // targets row t is the distribution over the move AFTER prefix tokens 0..t,
  // i.e. over the board reached by moves tokens[1..t].
  const train = corpus.map((tokens) => {
    const input = tokens.slice(0, -1);
    const targets = tokens.slice(1);
    let softQ: Float64Array | null = null;
    if (signal === 'solver') {
      softQ = new Float64Array(input.length * 10);
      for (let t = 0; t < input.length; t++) {
        const q9 = solverQ9(boardFromMoves(tokens.slice(1, t + 1).map((x) => x - 1)));
        for (let m = 0; m < 9; m++) softQ[t * 10 + (m + 1)] = q9[m];
      }
    }
    return { input, targets, softQ };
  });
  const evalSet = evalSubset(train, seed);

  const lossOf = (ex: (typeof train)[number]): Tensor => {
    const logits = model.forward(ex.input);
    return ex.softQ ? softCrossEntropyLoss(logits, ex.softQ) : crossEntropyLoss(logits, ex.targets);
  };

  const stepsOut: TicStep[] = [];
  const weights: Float64Array[] = [];
  for (let i = 0; i < steps; i++) {
    let sum = 0;
    for (const ex of evalSet) sum += lossOf(ex).data[0];
    stepsOut.push({ index: i, loss: sum / evalSet.length });
    weights.push(flatten(params));
    if (i === steps - 1) break;
    const ex = train[randInt(rng, train.length)];
    optimizer.zeroGrad();
    backward(lossOf(ex));
    l1Inject(params, l1Lambda);
    optimizer.step();
  }

  const replay = new MiniGPT(GPT_CFG, mulberry32(1));
  const at = (step: number, moves: number[]): ForwardViz => {
    load(replay.params(), weights[step]);
    replay.forward(gameIds(moves));
    return replay.viz!;
  };

  return {
    arch: 'gpt',
    signal,
    steps: stepsOut,
    weights,
    cfg: GPT_CFG,
    meta: { nHeads: GPT_CFG.nHeads, nUnits: GPT_CFG.ffnHid, paramCount: weights[0].length },
    probsAt: (s, m) => at(s, m).probs,
    unitsAt(s, m) {
      const viz = at(s, m);
      const [Tn, F] = viz.ffnHidden.shape;
      const row = new Float64Array(F);
      for (let u = 0; u < F; u++) row[u] = viz.ffnHidden.data[(Tn - 1) * F + u];
      return row;
    },
    vizAt: at,
    lensAt(s, m) {
      at(s, m);
      return logitLens(replay);
    },
    tokenTable(s) {
      load(replay.params(), weights[s]);
      return replay.tokenEmb.weight.snapshot();
    },
    sparsity: (s, thr) => ffnSparsity(replay.params(), weights[s], GPT_CFG.embDim, GPT_CFG.ffnHid, thr)
  };
}

// ---------------------------------------------------------------------------
// Board arms — positions
// ---------------------------------------------------------------------------

function trainBoard(
  arch: BoardArch,
  corpus: number[][],
  signal: Signal,
  rng: Rng,
  seed: number,
  steps: number,
  l1Lambda: number,
  teacher: TicRun | null,
  temperature: number
): TicRun {
  const model = makeBoardModel(arch, rng);
  const params = model.params();
  const optimizer = new Adam(params, LR);

  // Positions = every prefix of every game, with natural multiplicity — the
  // same effective data distribution the GPT arm sees through its sequences.
  interface Pos {
    moves: number[];
    cells: number[];
    target: number;
    q: Float64Array | null;
  }
  const teacherCache = new Map<number, Float64Array>();
  const positions: Pos[] = [];
  for (const tokens of corpus) {
    const game = tokens.slice(1).map((x) => x - 1);
    for (let k = 0; k < game.length; k++) {
      const moves = game.slice(0, k);
      const cells = boardFromMoves(moves) as number[];
      let q: Float64Array | null = null;
      if (signal === 'solver') q = solverQ9(cells as Board);
      else if (signal === 'distill') {
        const key = boardKey(cells as Board);
        q = teacherCache.get(key) ?? teacherQ9(teacher!, moves, temperature);
        teacherCache.set(key, q);
      }
      positions.push({ moves, cells, target: game[k], q });
    }
  }
  const evalSet = evalSubset(positions, seed);

  const lossOf = (p: Pos): Tensor => {
    const logits = model.forward(p.cells);
    return p.q ? softCrossEntropyLoss(logits, p.q) : crossEntropyLoss(logits, [p.target]);
  };

  const stepsOut: TicStep[] = [];
  const weights: Float64Array[] = [];
  for (let i = 0; i < steps; i++) {
    let sum = 0;
    for (const p of evalSet) sum += lossOf(p).data[0];
    stepsOut.push({ index: i, loss: sum / evalSet.length });
    weights.push(flatten(params));
    if (i === steps - 1) break;
    const p = positions[randInt(rng, positions.length)];
    optimizer.zeroGrad();
    backward(lossOf(p));
    l1Inject(params, l1Lambda);
    optimizer.step();
  }

  const replay = makeBoardModel(arch, mulberry32(1));
  const nUnits = arch === 'mlp' ? MLP_HID : ENC_FFN;
  const at = (step: number, moves: number[]): BoardViz => {
    load(replay.params(), weights[step]);
    replay.forward(boardFromMoves(moves) as number[]);
    return replay.viz!;
  };

  return {
    arch,
    signal,
    steps: stepsOut,
    weights,
    cfg: null,
    meta: { nHeads: arch === 'encoder' ? 2 : 0, nUnits, paramCount: weights[0].length },
    probsAt(s, m) {
      const out = new Float64Array(10);
      out.set(at(s, m).probs, 1); // slot 0 ('·') stays 0 — board archs can't emit it
      return out;
    },
    unitsAt(s, m) {
      const viz = at(s, m);
      const [rows, F] = viz.hidden.shape;
      const out = new Float64Array(F);
      for (let r = 0; r < rows; r++) for (let u = 0; u < F; u++) out[u] += viz.hidden.data[r * F + u] / rows;
      return out;
    },
    vizAt: at,
    lensAt: () => null,
    tokenTable(s) {
      load(replay.params(), weights[s]);
      if (arch === 'encoder') {
        const ce = replay.params()[1]; // cellEmb [9, d] — cell identity lives here
        return { data: new Float64Array(ce.data), shape: [...ce.shape] };
      }
      // MLP: per-cell identity lives in the input weights — take each cell's
      // X-state row of fc1 ([27,hid] row 3i+1) as that cell's vector.
      const w = replay.params()[0]; // fc1.w [27, MLP_HID]
      const data = new Float64Array(9 * MLP_HID);
      for (let c = 0; c < 9; c++) data.set(w.data.subarray((3 * c + 1) * MLP_HID, (3 * c + 2) * MLP_HID), c * MLP_HID);
      return { data, shape: [9, MLP_HID] };
    },
    sparsity: (s, thr) =>
      arch === 'mlp'
        ? ffnSparsity(replay.params(), weights[s], 27, MLP_HID, thr)
        : ffnSparsity(replay.params(), weights[s], 24, ENC_FFN, thr)
  };
}

/**
 * The circuits-repo readout, generalized: locate the fc1 [inDim, hid] and
 * fc2 [hid, outDim] segments in the flat vector by walking the param list's
 * shapes, then count surviving edges and live units at the threshold.
 */
function ffnSparsity(
  params: Tensor[],
  weights: Float64Array,
  inDim: number,
  hid: number,
  thr: number
): SparsityReport {
  let small = 0;
  for (let i = 0; i < weights.length; i++) if (Math.abs(weights[i]) < thr) small++;

  let off = 0;
  let fc1 = -1;
  let fc2 = -1;
  let outDim = 0;
  for (const p of params) {
    const [r, c] = p.shape.length === 2 ? p.shape : [1, p.shape[0]];
    if (r === inDim && c === hid && fc1 < 0) fc1 = off;
    else if (r === hid && fc2 < 0 && fc1 >= 0) {
      fc2 = off;
      outDim = c;
    }
    off += p.data.length;
  }
  if (fc1 < 0 || fc2 < 0) return { frac: small / weights.length, liveUnits: 0, edges: 0 };

  let edges = 0;
  let live = 0;
  for (let u = 0; u < hid; u++) {
    let inn = 0;
    let out = 0;
    for (let d = 0; d < inDim; d++) if (Math.abs(weights[fc1 + d * hid + u]) >= thr) inn++;
    for (let d = 0; d < outDim; d++) if (Math.abs(weights[fc2 + u * outDim + d]) >= thr) out++;
    edges += inn + out;
    if (inn > 0 && out > 0) live++;
  }
  return { frac: small / weights.length, liveUnits: live, edges };
}
