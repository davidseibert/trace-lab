/**
 * The two board-state architectures for the Tic·tac arch grid — both read the
 * POSITION (9 cells, values empty/X/O) instead of the move history, so unlike
 * the GPT arm neither has to make board state emerge; it's the input.
 *
 *  - BoardEncoder: a transformer ENCODER — per-cell tokens, no causal mask
 *    (softmaxRows), one pre-norm block, and a per-cell head SHARED across
 *    cells, so cell identity lives only in the position embeddings. This is
 *    the architecture where "the block move attends to the threatened line"
 *    is a literal, readable attention edge — including edges to EMPTY cells,
 *    which the move-sequence GPT cannot represent at all.
 *  - BoardMLP: one hidden layer over the 27-dim one-hot board. The null
 *    model: attention adds no capacity on a fully-observed 9-cell state, and
 *    this arm exists to measure what (if anything) it adds anyway.
 *
 * Both are parameter-matched to the GPT arm (~5.5k) and draw init from one
 * seeded Rng, so runs are deterministic like everything else in the lens.
 */

import {
  Tensor,
  addElem,
  matmul,
  relu,
  scale,
  softmaxRows,
  reshape,
  layerNorm,
  add,
  embeddingLookup,
  type TensorSnap
} from '../llm/tensor';
import type { Rng } from '../llm/rng';

export type BoardArch = 'encoder' | 'mlp';

export interface BoardViz {
  /** [9] raw logits, one per cell. */
  logits: Float64Array;
  /** [9] softmax of the logits. */
  probs: Float64Array;
  /** Post-ReLU hidden: mlp [1, 150] · encoder [9, 48]. */
  hidden: TensorSnap;
  /** Encoder only: [nHeads, 9, 9] cell-to-cell attention. */
  attn: TensorSnap | null;
  /** Encoder only: the cell position-embedding table [9, d]. */
  cellEmb: TensorSnap | null;
}

export interface BoardModel {
  arch: BoardArch;
  /** cells: 9 values 0 (empty) | 1 (X) | 2 (O) → [1, 9] logits tensor. */
  forward(cells: number[]): Tensor;
  params(): Tensor[];
  viz: BoardViz | null;
}

// Dimensions chosen to land near the GPT arm's ~5.7k params.
export const ENC_DIM = 24;
export const ENC_HEADS = 2;
export const ENC_FFN = 48;
export const MLP_HID = 150;

const softmax9 = (logits: Float64Array): Float64Array => {
  const out = new Float64Array(9);
  let mx = -Infinity;
  for (let i = 0; i < 9; i++) mx = Math.max(mx, logits[i]);
  let s = 0;
  for (let i = 0; i < 9; i++) {
    out[i] = Math.exp(logits[i] - mx);
    s += out[i];
  }
  for (let i = 0; i < 9; i++) out[i] /= s;
  return out;
};

// ---------------------------------------------------------------------------
// Shared small layers (mirroring model.ts's Linear/Embedding, which are
// module-private there)
// ---------------------------------------------------------------------------

class Lin {
  w: Tensor;
  b: Tensor;
  constructor(inDim: number, outDim: number, rng: Rng) {
    this.w = Tensor.randn([inDim, outDim], Math.sqrt(2 / (inDim + outDim)), rng);
    this.b = Tensor.zeros([outDim]);
  }
  forward(x: Tensor): Tensor {
    return add(matmul(x, this.w), this.b);
  }
  params(): Tensor[] {
    return [this.w, this.b];
  }
}

// [T, H·HD] ⇄ [H, T, HD] permutations with pass-through backwards — the same
// data movement as model.ts's private #splitHeads/#mergeHeads/#transpose.
function splitHeads(x: Tensor, T: number, H: number, HD: number): Tensor {
  const out = new Tensor(new Float64Array(x.data.length), [H, T, HD]);
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++)
      for (let d = 0; d < HD; d++) out.data[h * T * HD + t * HD + d] = x.data[t * H * HD + h * HD + d];
  out._children = [x];
  out._op = 'splitHeads';
  out._backward = () => {
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++)
        for (let d = 0; d < HD; d++) x.grad[t * H * HD + h * HD + d] += out.grad[h * T * HD + t * HD + d];
  };
  return out;
}

function mergeHeads(x: Tensor, T: number, H: number, HD: number): Tensor {
  const D = H * HD;
  const out = new Tensor(new Float64Array(T * D), [T, D]);
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++)
      for (let d = 0; d < HD; d++) out.data[t * D + h * HD + d] = x.data[h * T * HD + t * HD + d];
  out._children = [x];
  out._op = 'mergeHeads';
  out._backward = () => {
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++)
        for (let d = 0; d < HD; d++) x.grad[h * T * HD + t * HD + d] += out.grad[t * D + h * HD + d];
  };
  return out;
}

function transpose3(x: Tensor, B: number, T: number, HD: number): Tensor {
  const out = new Tensor(new Float64Array(x.data.length), [B, HD, T]);
  for (let b = 0; b < B; b++)
    for (let t = 0; t < T; t++)
      for (let d = 0; d < HD; d++) out.data[b * HD * T + d * T + t] = x.data[b * T * HD + t * HD + d];
  out._children = [x];
  out._op = 'transpose';
  out._backward = () => {
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++)
        for (let d = 0; d < HD; d++) x.grad[b * T * HD + t * HD + d] += out.grad[b * HD * T + d * T + t];
  };
  return out;
}

// ---------------------------------------------------------------------------
// BoardEncoder
// ---------------------------------------------------------------------------

class BoardEncoder implements BoardModel {
  arch = 'encoder' as const;
  viz: BoardViz | null = null;

  stateEmb: Tensor; // [3, d] — empty / X / O
  cellEmb: Tensor; // [9, d] — the ONLY place cell identity lives
  ln1: { g: Tensor; b: Tensor };
  ln2: { g: Tensor; b: Tensor };
  q: Lin;
  k: Lin;
  v: Lin;
  o: Lin;
  fc1: Lin;
  fc2: Lin;
  head: Lin; // [d, 1], shared across cells

  constructor(rng: Rng) {
    const d = ENC_DIM;
    this.stateEmb = Tensor.randn([3, d], 0.1, rng);
    this.cellEmb = Tensor.randn([9, d], 0.1, rng);
    this.ln1 = { g: Tensor.filled([d], 1), b: Tensor.zeros([d]) };
    this.ln2 = { g: Tensor.filled([d], 1), b: Tensor.zeros([d]) };
    this.q = new Lin(d, d, rng);
    this.k = new Lin(d, d, rng);
    this.v = new Lin(d, d, rng);
    this.o = new Lin(d, d, rng);
    this.fc1 = new Lin(d, ENC_FFN, rng);
    this.fc2 = new Lin(ENC_FFN, d, rng);
    this.head = new Lin(d, 1, rng);
  }

  forward(cells: number[]): Tensor {
    const d = ENC_DIM;
    const H = ENC_HEADS;
    const HD = d / H;

    // Per-cell input: what's here + which cell this is.
    const x0 = addElem(embeddingLookup(this.stateEmb, cells), embeddingLookup(this.cellEmb, [...Array(9).keys()]));

    // Pre-norm block, bidirectional: every cell sees every cell — including
    // the empty ones, which is exactly what the GPT arm cannot do.
    const n1 = layerNorm(x0, this.ln1.g, this.ln1.b);
    const qr = splitHeads(this.q.forward(n1), 9, H, HD);
    const kr = splitHeads(this.k.forward(n1), 9, H, HD);
    const vr = splitHeads(this.v.forward(n1), 9, H, HD);
    const attn = softmaxRows(scale(matmul(qr, transpose3(kr, H, 9, HD)), 1 / Math.sqrt(HD)));
    const attnOut = this.o.forward(mergeHeads(matmul(attn, vr), 9, H, HD));
    const res1 = addElem(x0, attnOut);

    const n2 = layerNorm(res1, this.ln2.g, this.ln2.b);
    const hid = relu(this.fc1.forward(n2));
    const out = addElem(res1, this.fc2.forward(hid));

    // Shared readout: logit_i = w·h_i (+b) — [9,1] reshaped to [1,9].
    const logits = reshape(this.head.forward(out), [1, 9]);

    const flat = new Float64Array(logits.data);
    this.viz = {
      logits: flat,
      probs: softmax9(flat),
      hidden: hid.snapshot(),
      attn: attn.snapshot(),
      cellEmb: this.cellEmb.snapshot()
    };
    return logits;
  }

  params(): Tensor[] {
    return [
      this.stateEmb,
      this.cellEmb,
      this.ln1.g,
      this.ln1.b,
      ...this.q.params(),
      ...this.k.params(),
      ...this.v.params(),
      ...this.o.params(),
      this.ln2.g,
      this.ln2.b,
      ...this.fc1.params(),
      ...this.fc2.params(),
      ...this.head.params()
    ];
  }
}

// ---------------------------------------------------------------------------
// BoardMLP
// ---------------------------------------------------------------------------

class BoardMLP implements BoardModel {
  arch = 'mlp' as const;
  viz: BoardViz | null = null;
  fc1: Lin;
  fc2: Lin;

  constructor(rng: Rng) {
    this.fc1 = new Lin(27, MLP_HID, rng);
    this.fc2 = new Lin(MLP_HID, 9, rng);
  }

  forward(cells: number[]): Tensor {
    // One-hot 27: cell i contributes a 1 at 3i + state.
    const oneHot = Tensor.zeros([1, 27]);
    for (let i = 0; i < 9; i++) oneHot.data[3 * i + cells[i]] = 1;
    const hid = relu(this.fc1.forward(oneHot));
    const logits = this.fc2.forward(hid); // [1, 9]

    const flat = new Float64Array(logits.data);
    this.viz = { logits: flat, probs: softmax9(flat), hidden: hid.snapshot(), attn: null, cellEmb: null };
    return logits;
  }

  params(): Tensor[] {
    return [...this.fc1.params(), ...this.fc2.params()];
  }
}

export function makeBoardModel(arch: BoardArch, rng: Rng): BoardModel {
  return arch === 'encoder' ? new BoardEncoder(rng) : new BoardMLP(rng);
}

export function paramCount(model: BoardModel): number {
  return model.params().reduce((s, p) => s + p.data.length, 0);
}
