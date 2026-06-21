/**
 * A hand-built mini-GPT: token + position embeddings, one pre-norm transformer
 * block (multi-head causal self-attention + ReLU feed-forward, both with
 * residuals), a final layer norm, and a linear head to vocabulary logits.
 *
 * Every weight is a `Tensor`, so the whole forward pass is differentiable via
 * `tensor.ts`'s autograd and trainable with plain Adam. The model is tiny by
 * design (embedding dims of 8–16) — small enough that one training step is
 * sub-millisecond and the entire run can be precomputed into a trace.
 *
 * `forward()` records a `ForwardViz` snapshot of every intermediate activation;
 * that snapshot is what the UI panels render.
 */

import type { Rng } from './rng';
import {
  Tensor,
  type TensorSnap,
  matmul,
  add,
  addElem,
  relu,
  scale,
  maskedSoftmax,
  layerNorm,
  embeddingLookup,
  softmaxVec
} from './tensor';

export interface ModelConfig {
  vocabSize: number;
  embDim: number;
  ctxLen: number;
  nHeads: number;
  ffnHid: number;
}

/** Frozen snapshot of one forward pass — the data every LLM panel reads. */
export interface ForwardViz {
  tokenIds: number[];
  /** [T, embDim] token embeddings. */
  tokEmb: TensorSnap;
  /** [T, embDim] position embeddings. */
  posEmb: TensorSnap;
  /** [T, embDim] token + position. */
  embedded: TensorSnap;
  /** [nHeads, T, T] causal attention weights. */
  attn: TensorSnap;
  /** [T, ffnHid] post-ReLU hidden activations. */
  ffnHidden: TensorSnap;
  /** [T, embDim] block output. */
  blockOut: TensorSnap;
  /** [T, vocabSize] raw logits. */
  logits: TensorSnap;
  /** [vocabSize] next-token distribution (softmax of the last position). */
  probs: Float64Array;
}

interface Layer {
  params(): Tensor[];
}

class Embedding implements Layer {
  weight: Tensor;
  constructor(rows: number, dim: number, std: number, rng: Rng) {
    this.weight = Tensor.randn([rows, dim], std, rng);
  }
  forward(indices: number[]): Tensor {
    return embeddingLookup(this.weight, indices);
  }
  params(): Tensor[] {
    return [this.weight];
  }
}

class Linear implements Layer {
  weight: Tensor;
  bias: Tensor;
  constructor(inDim: number, outDim: number, rng: Rng) {
    this.weight = Tensor.randn([inDim, outDim], Math.sqrt(2 / (inDim + outDim)), rng);
    this.bias = Tensor.zeros([outDim]);
  }
  forward(x: Tensor): Tensor {
    return add(matmul(x, this.weight), this.bias);
  }
  params(): Tensor[] {
    return [this.weight, this.bias];
  }
}

class LayerNorm implements Layer {
  gamma: Tensor;
  beta: Tensor;
  constructor(dim: number) {
    this.gamma = Tensor.filled([dim], 1);
    this.beta = Tensor.zeros([dim]);
  }
  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gamma, this.beta);
  }
  params(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

class MultiHeadAttention implements Layer {
  dim: number;
  nHeads: number;
  headDim: number;
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  outProj: Linear;
  /** [nHeads, T, T] attention weights captured on the last forward. */
  lastAttn: TensorSnap | null = null;

  constructor(dim: number, nHeads: number, rng: Rng) {
    this.dim = dim;
    this.nHeads = nHeads;
    this.headDim = dim / nHeads;
    this.qProj = new Linear(dim, dim, rng);
    this.kProj = new Linear(dim, dim, rng);
    this.vProj = new Linear(dim, dim, rng);
    this.outProj = new Linear(dim, dim, rng);
  }

  forward(x: Tensor): Tensor {
    const T = x.shape[0];
    const H = this.nHeads;
    const HD = this.headDim;

    const Q = this.qProj.forward(x);
    const K = this.kProj.forward(x);
    const V = this.vProj.forward(x);

    const qr = this.#splitHeads(Q, T, H, HD); // [H, T, HD]
    const kr = this.#splitHeads(K, T, H, HD);
    const vr = this.#splitHeads(V, T, H, HD);

    const kt = this.#transpose(kr, H, T, HD); // [H, HD, T]
    const scores = scale(matmul(qr, kt), 1 / Math.sqrt(HD)); // [H, T, T]
    const attn = maskedSoftmax(scores, T);
    this.lastAttn = attn.snapshot();

    const attnOut = matmul(attn, vr); // [H, T, HD]
    const concat = this.#mergeHeads(attnOut, T, H, HD); // [T, D]
    return this.outProj.forward(concat);
  }

  /** [T, H*HD] -> [H, T, HD] */
  #splitHeads(x: Tensor, T: number, H: number, HD: number): Tensor {
    const out = new Tensor(new Float64Array(x.data.length), [H, T, HD]);
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++)
        for (let d = 0; d < HD; d++)
          out.data[h * T * HD + t * HD + d] = x.data[t * H * HD + h * HD + d];
    out._children = [x];
    out._op = 'splitHeads';
    out._backward = () => {
      for (let t = 0; t < T; t++)
        for (let h = 0; h < H; h++)
          for (let d = 0; d < HD; d++)
            x.grad[t * H * HD + h * HD + d] += out.grad[h * T * HD + t * HD + d];
    };
    return out;
  }

  /** [H, T, HD] -> [T, H*HD] */
  #mergeHeads(x: Tensor, T: number, H: number, HD: number): Tensor {
    const D = H * HD;
    const out = new Tensor(new Float64Array(T * D), [T, D]);
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++)
        for (let d = 0; d < HD; d++)
          out.data[t * D + h * HD + d] = x.data[h * T * HD + t * HD + d];
    out._children = [x];
    out._op = 'mergeHeads';
    out._backward = () => {
      for (let t = 0; t < T; t++)
        for (let h = 0; h < H; h++)
          for (let d = 0; d < HD; d++)
            x.grad[h * T * HD + t * HD + d] += out.grad[t * D + h * HD + d];
    };
    return out;
  }

  /** [B, T, HD] -> [B, HD, T] */
  #transpose(x: Tensor, B: number, T: number, HD: number): Tensor {
    const out = new Tensor(new Float64Array(x.data.length), [B, HD, T]);
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++)
        for (let d = 0; d < HD; d++)
          out.data[b * HD * T + d * T + t] = x.data[b * T * HD + t * HD + d];
    out._children = [x];
    out._op = 'transpose';
    out._backward = () => {
      for (let b = 0; b < B; b++)
        for (let t = 0; t < T; t++)
          for (let d = 0; d < HD; d++)
            x.grad[b * T * HD + t * HD + d] += out.grad[b * HD * T + d * T + t];
    };
    return out;
  }

  params(): Tensor[] {
    return [
      ...this.qProj.params(),
      ...this.kProj.params(),
      ...this.vProj.params(),
      ...this.outProj.params()
    ];
  }
}

class FeedForward implements Layer {
  fc1: Linear;
  fc2: Linear;
  /** [T, ffnHid] post-ReLU hidden, captured on the last forward. */
  lastHidden: TensorSnap | null = null;

  constructor(dim: number, hidDim: number, rng: Rng) {
    this.fc1 = new Linear(dim, hidDim, rng);
    this.fc2 = new Linear(hidDim, dim, rng);
  }
  forward(x: Tensor): Tensor {
    const h = relu(this.fc1.forward(x));
    this.lastHidden = h.snapshot();
    return this.fc2.forward(h);
  }
  params(): Tensor[] {
    return [...this.fc1.params(), ...this.fc2.params()];
  }
}

class TransformerBlock implements Layer {
  attn: MultiHeadAttention;
  ffn: FeedForward;
  ln1: LayerNorm;
  ln2: LayerNorm;

  constructor(dim: number, nHeads: number, ffnHid: number, rng: Rng) {
    this.attn = new MultiHeadAttention(dim, nHeads, rng);
    this.ffn = new FeedForward(dim, ffnHid, rng);
    this.ln1 = new LayerNorm(dim);
    this.ln2 = new LayerNorm(dim);
  }
  forward(x: Tensor): Tensor {
    const res1 = addElem(x, this.attn.forward(this.ln1.forward(x)));
    return addElem(res1, this.ffn.forward(this.ln2.forward(res1)));
  }
  params(): Tensor[] {
    return [...this.attn.params(), ...this.ffn.params(), ...this.ln1.params(), ...this.ln2.params()];
  }
}

export class MiniGPT {
  cfg: ModelConfig;
  tokenEmb: Embedding;
  posEmb: Embedding;
  block: TransformerBlock;
  lnFinal: LayerNorm;
  head: Linear;
  /** Snapshot of the most recent forward pass. */
  viz: ForwardViz | null = null;

  constructor(cfg: ModelConfig, rng: Rng) {
    this.cfg = cfg;
    this.tokenEmb = new Embedding(cfg.vocabSize, cfg.embDim, 0.1, rng);
    this.posEmb = new Embedding(cfg.ctxLen, cfg.embDim, 0.1, rng);
    this.block = new TransformerBlock(cfg.embDim, cfg.nHeads, cfg.ffnHid, rng);
    this.lnFinal = new LayerNorm(cfg.embDim);
    this.head = new Linear(cfg.embDim, cfg.vocabSize, rng);
  }

  /** Run the model, returning logits [T, V] and recording `this.viz`. */
  forward(tokenIds: number[]): Tensor {
    const T = tokenIds.length;
    const V = this.cfg.vocabSize;
    const posIds = Array.from({ length: T }, (_, i) => i);

    const tokEmb = this.tokenEmb.forward(tokenIds);
    const posEmb = this.posEmb.forward(posIds);
    const embedded = addElem(tokEmb, posEmb);

    const blockOut = this.block.forward(embedded);
    const normed = this.lnFinal.forward(blockOut);
    const logits = this.head.forward(normed);

    const lastLogits = new Float64Array(V);
    for (let v = 0; v < V; v++) lastLogits[v] = logits.data[(T - 1) * V + v];

    this.viz = {
      tokenIds: [...tokenIds],
      tokEmb: tokEmb.snapshot(),
      posEmb: posEmb.snapshot(),
      embedded: embedded.snapshot(),
      attn: this.block.attn.lastAttn!,
      ffnHidden: this.block.ffn.lastHidden!,
      blockOut: blockOut.snapshot(),
      logits: logits.snapshot(),
      probs: softmaxVec(lastLogits)
    };

    return logits;
  }

  params(): Tensor[] {
    return [
      ...this.tokenEmb.params(),
      ...this.posEmb.params(),
      ...this.block.params(),
      ...this.lnFinal.params(),
      ...this.head.params()
    ];
  }
}

/** Adam optimiser over a flat list of parameter tensors. */
export class Adam {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  t = 0;
  #m: Float64Array[];
  #v: Float64Array[];

  constructor(params: Tensor[], lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.#m = params.map((p) => new Float64Array(p.data.length));
    this.#v = params.map((p) => new Float64Array(p.data.length));
  }

  step(): void {
    this.t++;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      const m = this.#m[pi];
      const v = this.#v[pi];
      for (let i = 0; i < p.data.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * p.grad[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * p.grad[i] * p.grad[i];
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}
