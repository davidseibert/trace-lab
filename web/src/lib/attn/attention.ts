/**
 * Static scaled dot-product attention over small, literal matrices — no
 * training, no autograd, just the forward arithmetic. Every stage the
 * mechanism touches (Q/K/V projections, raw scores, scaling, softmax,
 * weighted sum, head concat, output projection) is captured as a plain
 * `TensorSnap` so the UI can render every number, not just the result.
 *
 * Reuses `tensor.ts`'s `Tensor`/`matmul`/`scale`/`softmaxVec` for numerical
 * consistency with the rest of the LLM lens, but never calls `backward()` —
 * there's nothing to train here.
 */

import { Tensor, type TensorSnap, matmul, scale, softmaxVec } from '../llm/tensor';
import type { Rng } from '../llm/rng';
import { gaussian } from '../llm/rng';

export interface AttnConfig {
  dModel: number;
  nHeads: number;
  causal: boolean;
}

export interface HeadTrace {
  head: number;
  Q: TensorSnap; // [T, headDim]
  K: TensorSnap; // [T, headDim]
  V: TensorSnap; // [T, headDim]
  scores: TensorSnap; // [T, T]  Q · Kᵀ
  scaled: TensorSnap; // [T, T]  scores / √headDim
  weights: TensorSnap; // [T, T]  softmax(scaled), causal-masked rows sum to 1 over cols ≤ row
  output: TensorSnap; // [T, headDim]  weights · V
}

export interface AttnTrace {
  cfg: AttnConfig;
  T: number;
  headDim: number;
  heads: HeadTrace[];
  concat: TensorSnap; // [T, dModel]  heads' outputs side by side
  output: TensorSnap; // [T, dModel]  concat · Wo
}

function toTensor(snap: TensorSnap): Tensor {
  return new Tensor(new Float64Array(snap.data), [...snap.shape]);
}

function transpose2d(t: Tensor): Tensor {
  const [R, C] = t.shape;
  const out = Tensor.zeros([C, R]);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out.data[c * R + r] = t.data[r * C + c];
  return out;
}

function sliceCols(t: Tensor, start: number, width: number): Tensor {
  const [R, C] = t.shape;
  const out = Tensor.zeros([R, width]);
  for (let r = 0; r < R; r++) for (let c = 0; c < width; c++) out.data[r * width + c] = t.data[r * C + start + c];
  return out;
}

function concatCols(mats: Tensor[]): Tensor {
  const R = mats[0].shape[0];
  const width = mats.reduce((s, m) => s + m.shape[1], 0);
  const out = Tensor.zeros([R, width]);
  let off = 0;
  for (const m of mats) {
    const w = m.shape[1];
    for (let r = 0; r < R; r++) for (let c = 0; c < w; c++) out.data[r * width + off + c] = m.data[r * w + c];
    off += w;
  }
  return out;
}

/**
 * Row-wise softmax. When `causal`, row `r` only sees columns `0..r` (the
 * standard "can't attend to the future" mask) — later columns are left at
 * their `Tensor.zeros` default rather than materialising a `-Infinity` mask,
 * so there's no sentinel value that could leak into the UI's color scaling.
 */
function computeWeights(scaled: Tensor, causal: boolean): Tensor {
  const T = scaled.shape[0];
  const weights = Tensor.zeros([T, T]);
  for (let r = 0; r < T; r++) {
    const width = causal ? r + 1 : T;
    const row = new Float64Array(width);
    for (let c = 0; c < width; c++) row[c] = scaled.data[r * T + c];
    weights.data.set(softmaxVec(row), r * T);
  }
  return weights;
}

export function computeAttention(
  cfg: AttnConfig,
  Xs: TensorSnap,
  Wqs: TensorSnap,
  Wks: TensorSnap,
  Wvs: TensorSnap,
  Wos: TensorSnap
): AttnTrace {
  const X = toTensor(Xs);
  const T = X.shape[0];
  const { dModel, nHeads } = cfg;
  const headDim = dModel / nHeads;

  const Qfull = matmul(X, toTensor(Wqs));
  const Kfull = matmul(X, toTensor(Wks));
  const Vfull = matmul(X, toTensor(Wvs));

  const heads: HeadTrace[] = [];
  const outs: Tensor[] = [];
  for (let h = 0; h < nHeads; h++) {
    const Q = sliceCols(Qfull, h * headDim, headDim);
    const K = sliceCols(Kfull, h * headDim, headDim);
    const V = sliceCols(Vfull, h * headDim, headDim);
    const scores = matmul(Q, transpose2d(K));
    const scaled = scale(scores, 1 / Math.sqrt(headDim));
    const weights = computeWeights(scaled, cfg.causal);
    const output = matmul(weights, V);
    heads.push({
      head: h,
      Q: Q.snapshot(),
      K: K.snapshot(),
      V: V.snapshot(),
      scores: scores.snapshot(),
      scaled: scaled.snapshot(),
      weights: weights.snapshot(),
      output: output.snapshot()
    });
    outs.push(output);
  }

  const concat = concatCols(outs);
  const output = matmul(concat, toTensor(Wos));

  return { cfg, T, headDim, heads, concat: concat.snapshot(), output: output.snapshot() };
}

/** Small, hand-checkable random values — Gaussian, rounded to one decimal. */
export function randomMatrix(rows: number, cols: number, rng: Rng, std = 0.6): TensorSnap {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = Math.round(gaussian(rng, std) * 10) / 10;
  return { data, shape: [rows, cols] };
}

export function identityMatrix(n: number): TensorSnap {
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) data[i * n + i] = 1;
  return { data, shape: [n, n] };
}
