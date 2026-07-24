/**
 * A minimal reverse-mode autograd engine over dense Float64 tensors — enough to
 * build and train a tiny transformer by hand, with no ML framework.
 *
 * Each `Tensor` records the operation that produced it and a `_backward` closure
 * that pushes gradients to its inputs. `backward()` does a topological sort from
 * the loss and runs those closures in reverse. This is the same idea as
 * Karpathy's micrograd, but vectorised over flat typed arrays so a whole
 * attention block is a handful of ops rather than thousands of scalar nodes.
 *
 * Shapes are plain `number[]`; data is row-major in a single `Float64Array`.
 */

import type { Rng } from './rng';
import { gaussian } from './rng';

let tensorIdCounter = 0;

export class Tensor {
  readonly id: number;
  data: Float64Array;
  shape: number[];
  grad: Float64Array;
  _backward: () => void = () => {};
  _children: Tensor[];
  _op: string;

  constructor(data: Float64Array, shape: number[], children: Tensor[] = [], op = '') {
    this.id = tensorIdCounter++;
    this.data = data;
    this.shape = shape;
    this.grad = new Float64Array(this.data.length);
    this._children = children;
    this._op = op;
  }

  static zeros(shape: number[]): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float64Array(n), shape);
  }

  static filled(shape: number[], value: number): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float64Array(n).fill(value), shape);
  }

  /** Gaussian-initialised tensor, drawn from the supplied seeded stream. */
  static randn(shape: number[], std: number, rng: Rng): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = gaussian(rng, std);
    return new Tensor(d, shape);
  }

  /** Immutable copy of the current values + shape, for trace snapshots. */
  snapshot(): TensorSnap {
    return { data: new Float64Array(this.data), shape: [...this.shape] };
  }

  get size(): number {
    return this.data.length;
  }
}

/** A plain, frozen-in-time copy of a tensor's values — what the UI renders. */
export interface TensorSnap {
  data: Float64Array;
  shape: number[];
}

// ---------------------------------------------------------------------------
// Operations (each wires up its own backward pass)
// ---------------------------------------------------------------------------

/** Matrix multiply. Supports 2D × 2D and batched 3D × 3D (shared batch dim). */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length === 2 && b.shape.length === 2) {
    const [M, K] = a.shape;
    const N = b.shape[1];
    const out = Tensor.zeros([M, N]);
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += a.data[i * K + k] * b.data[k * N + j];
        out.data[i * N + j] = s;
      }
    out._children = [a, b];
    out._op = 'matmul';
    out._backward = () => {
      for (let i = 0; i < M; i++)
        for (let k = 0; k < K; k++) {
          let s = 0;
          for (let j = 0; j < N; j++) s += out.grad[i * N + j] * b.data[k * N + j];
          a.grad[i * K + k] += s;
        }
      for (let k = 0; k < K; k++)
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let i = 0; i < M; i++) s += a.data[i * K + k] * out.grad[i * N + j];
          b.grad[k * N + j] += s;
        }
    };
    return out;
  }

  // Batched 3D: [B,M,K] × [B,K,N] -> [B,M,N]
  const B = a.shape[0];
  const M = a.shape[1];
  const K = a.shape[2];
  const N = b.shape[2];
  const out = Tensor.zeros([B, M, N]);
  for (let bb = 0; bb < B; bb++) {
    const aOff = bb * M * K;
    const bOff = bb * K * N;
    const oOff = bb * M * N;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += a.data[aOff + i * K + k] * b.data[bOff + k * N + j];
        out.data[oOff + i * N + j] = s;
      }
  }
  out._children = [a, b];
  out._op = 'matmul3d';
  out._backward = () => {
    for (let bb = 0; bb < B; bb++) {
      const aOff = bb * M * K;
      const bOff = bb * K * N;
      const oOff = bb * M * N;
      for (let i = 0; i < M; i++)
        for (let k = 0; k < K; k++) {
          let s = 0;
          for (let j = 0; j < N; j++) s += out.grad[oOff + i * N + j] * b.data[bOff + k * N + j];
          a.grad[aOff + i * K + k] += s;
        }
      for (let k = 0; k < K; k++)
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let i = 0; i < M; i++) s += a.data[aOff + i * K + k] * out.grad[oOff + i * N + j];
          b.grad[bOff + k * N + j] += s;
        }
    }
  };
  return out;
}

/** Add a row-vector bias `b` to every row of `a`. */
export function add(a: Tensor, b: Tensor): Tensor {
  const out = new Tensor(new Float64Array(a.data), [...a.shape]);
  const N = b.data.length;
  const M = a.data.length / N;
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out.data[i * N + j] += b.data[j];
  out._children = [a, b];
  out._op = 'add';
  out._backward = () => {
    for (let i = 0; i < a.data.length; i++) a.grad[i] += out.grad[i];
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) b.grad[j] += out.grad[i * N + j];
  };
  return out;
}

/** Elementwise add of two same-shape tensors (residual connections). */
export function addElem(a: Tensor, b: Tensor): Tensor {
  const out = new Tensor(new Float64Array(a.data.length), [...a.shape]);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  out._children = [a, b];
  out._op = 'addElem';
  out._backward = () => {
    for (let i = 0; i < a.data.length; i++) {
      a.grad[i] += out.grad[i];
      b.grad[i] += out.grad[i];
    }
  };
  return out;
}

export function relu(a: Tensor): Tensor {
  const out = new Tensor(new Float64Array(a.data.length), [...a.shape]);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] > 0 ? a.data[i] : 0;
  out._children = [a];
  out._op = 'relu';
  out._backward = () => {
    for (let i = 0; i < a.data.length; i++) a.grad[i] += (out.data[i] > 0 ? 1 : 0) * out.grad[i];
  };
  return out;
}

/** Multiply every element by a constant. */
export function scale(a: Tensor, s: number): Tensor {
  const out = new Tensor(new Float64Array(a.data.length), [...a.shape]);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * s;
  out._children = [a];
  out._op = 'scale';
  out._backward = () => {
    for (let i = 0; i < a.data.length; i++) a.grad[i] += out.grad[i] * s;
  };
  return out;
}

/**
 * Row-wise softmax over the last `T` columns with a CAUSAL mask: position t can
 * only attend to positions ≤ t. `a` is treated as a stack of rows of width T.
 */
export function maskedSoftmax(a: Tensor, T: number): Tensor {
  const totalRows = a.data.length / T;
  const out = new Tensor(new Float64Array(a.data.length), [...a.shape]);
  const masked = new Float64Array(a.data);
  for (let r = 0; r < totalRows; r++) {
    const rowInBlock = r % T;
    const off = r * T;
    for (let c = rowInBlock + 1; c < T; c++) masked[off + c] = -1e9;
  }
  for (let r = 0; r < totalRows; r++) {
    const off = r * T;
    let mx = -Infinity;
    for (let c = 0; c < T; c++) mx = Math.max(mx, masked[off + c]);
    let s = 0;
    for (let c = 0; c < T; c++) {
      out.data[off + c] = Math.exp(masked[off + c] - mx);
      s += out.data[off + c];
    }
    for (let c = 0; c < T; c++) out.data[off + c] /= s;
  }
  out._children = [a];
  out._op = 'maskedSoftmax';
  out._backward = () => {
    for (let r = 0; r < totalRows; r++) {
      const off = r * T;
      const rowInBlock = r % T;
      for (let i = 0; i <= rowInBlock; i++) {
        let dot = 0;
        for (let j = 0; j <= rowInBlock; j++) dot += out.data[off + j] * out.grad[off + j];
        a.grad[off + i] += out.data[off + i] * (out.grad[off + i] - dot);
      }
    }
  };
  return out;
}

/** Per-row layer normalisation with learned scale (gamma) and shift (beta). */
export function layerNorm(a: Tensor, gamma: Tensor, beta: Tensor): Tensor {
  const T = a.shape[0];
  const D = a.shape[1];
  const out = new Tensor(new Float64Array(a.data.length), [T, D]);
  const means = new Float64Array(T);
  const vars = new Float64Array(T);
  const eps = 1e-5;

  for (let t = 0; t < T; t++) {
    const off = t * D;
    let mu = 0;
    for (let d = 0; d < D; d++) mu += a.data[off + d];
    mu /= D;
    means[t] = mu;
    let v = 0;
    for (let d = 0; d < D; d++) v += (a.data[off + d] - mu) ** 2;
    v /= D;
    vars[t] = v;
    const inv = 1 / Math.sqrt(v + eps);
    for (let d = 0; d < D; d++) {
      const norm = (a.data[off + d] - mu) * inv;
      out.data[off + d] = norm * gamma.data[d] + beta.data[d];
    }
  }

  out._children = [a, gamma, beta];
  out._op = 'layerNorm';
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const off = t * D;
      const mu = means[t];
      const v = vars[t];
      const inv = 1 / Math.sqrt(v + eps);
      for (let d = 0; d < D; d++) {
        const xnorm = (a.data[off + d] - mu) * inv;
        gamma.grad[d] += xnorm * out.grad[off + d];
        beta.grad[d] += out.grad[off + d];
      }
      let dxnormSum = 0;
      let dxnormXSum = 0;
      for (let d = 0; d < D; d++) {
        const dxnorm = out.grad[off + d] * gamma.data[d];
        dxnormSum += dxnorm;
        dxnormXSum += dxnorm * (a.data[off + d] - mu);
      }
      for (let d = 0; d < D; d++) {
        const dxnorm = out.grad[off + d] * gamma.data[d];
        a.grad[off + d] +=
          (1 / D) * inv * (D * dxnorm - dxnormSum - (a.data[off + d] - mu) * inv * inv * dxnormXSum);
      }
    }
  };
  return out;
}

/** Gather rows of `weight` by integer `indices` (token / position embedding). */
export function embeddingLookup(weight: Tensor, indices: number[]): Tensor {
  const T = indices.length;
  const D = weight.shape[1];
  const out = new Tensor(new Float64Array(T * D), [T, D]);
  for (let t = 0; t < T; t++) {
    const idx = indices[t];
    for (let d = 0; d < D; d++) out.data[t * D + d] = weight.data[idx * D + d];
  }
  out._children = [weight];
  out._op = 'embedding';
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const idx = indices[t];
      for (let d = 0; d < D; d++) weight.grad[idx * D + d] += out.grad[t * D + d];
    }
  };
  return out;
}

/**
 * Mean cross-entropy over `T` positions. Softmax is fused in here (numerically
 * stable) so the backward is the clean `prob - onehot` form.
 */
export function crossEntropyLoss(logits: Tensor, targets: number[]): Tensor {
  const T = logits.shape[0];
  const V = logits.shape[1];
  const probs = new Float64Array(logits.data.length);
  let loss = 0;

  for (let t = 0; t < T; t++) {
    const off = t * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) mx = Math.max(mx, logits.data[off + v]);
    let s = 0;
    for (let v = 0; v < V; v++) {
      probs[off + v] = Math.exp(logits.data[off + v] - mx);
      s += probs[off + v];
    }
    for (let v = 0; v < V; v++) probs[off + v] /= s;
    loss -= Math.log(probs[off + targets[t]] + 1e-10);
  }
  loss /= T;

  const out = new Tensor(new Float64Array([loss]), [1]);
  out._children = [logits];
  out._op = 'crossEntropy';
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const off = t * V;
      for (let v = 0; v < V; v++) {
        const indicator = v === targets[t] ? 1 : 0;
        logits.grad[off + v] += (probs[off + v] - indicator) / T;
      }
    }
  };
  return out;
}

/**
 * Reverse-mode sweep: zero grads, seed the root, run backward in topo order.
 * `seedIndex` picks which element of the root to differentiate (default 0, the
 * scalar-loss case). Seeding one element of an intermediate tensor — e.g. one
 * logit — turns a sweep into one row of a Jacobian, which is how the logit-lens
 * J-transport is computed exactly.
 */
export function backward(root: Tensor, seedIndex = 0): void {
  const visited = new Set<number>();
  const order: Tensor[] = [];
  function topo(node: Tensor) {
    if (visited.has(node.id)) return;
    visited.add(node.id);
    for (const c of node._children) topo(c);
    order.push(node);
  }
  topo(root);
  for (const node of order) node.grad.fill(0);
  root.grad[seedIndex] = 1;
  for (let i = order.length - 1; i >= 0; i--) order[i]._backward();
}

/** Stable softmax of a plain array (for reading off a probability vector). */
export function softmaxVec(values: Float64Array | number[]): Float64Array {
  const n = values.length;
  const out = new Float64Array(n);
  let mx = -Infinity;
  for (let i = 0; i < n; i++) mx = Math.max(mx, values[i]);
  let s = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(values[i] - mx);
    s += out[i];
  }
  for (let i = 0; i < n; i++) out[i] /= s;
  return out;
}
