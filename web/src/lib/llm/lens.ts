/**
 * logitLens — per-rung readout of the residual stream, in bits.
 *
 * The one-block mini-GPT has a three-rung residual ladder at the last position:
 *
 *   emb  = token + position          (what the prompt says)
 *   attn = emb + attention output    (what context has been mixed in)
 *   ffn  = attn + FFN output         (what memory has been recalled) — the final state
 *
 * The classic logit lens decodes each rung as if it were the final state: apply
 * the final LayerNorm + unembed and read a distribution. That's the READ
 * operation applied early — and reporting `−log₂ p` per rung turns "the
 * prediction sharpens with depth" into the literal MDL statement "the code for
 * the next token shortens with depth".
 *
 * The J-lens fixes the classic lens's known unfaithfulness: an early residual is
 * *not* in the final basis — the remaining layers still transform it. Because
 * this model is tiny, we compute the *exact* Jacobian `J = ∂ffn/∂rung` (one
 * reverse-mode sweep per output dim seeds one row) and decode `J·h` instead of
 * `h`: the rung's content transported by the map the network actually applies
 * to it. For the last rung J = I and the two lenses agree by construction.
 *
 * Each rung also gets a *visibility* fraction: how much of ‖h‖² lies in the row
 * space of `∂logits/∂h` (centered over the vocab, since softmax ignores a
 * uniform logit shift). The remainder is blind directions — components of the
 * residual that carry exactly 0 bits about the next token.
 */

import type { MiniGPT } from './model';
import { backward, softmaxVec } from './tensor';

export interface LensRung {
  key: 'emb' | 'attn' | 'ffn';
  label: string;
  /** [D] residual at the last position. */
  resid: Float64Array;
  /** Classic logit lens: decode `resid` straight through lnFinal + unembed. */
  naiveProbs: Float64Array;
  /** J-lens: decode `J·resid` — the rung transported into the final basis. */
  jProbs: Float64Array;
  /** Fraction of ‖resid‖² visible to the output (1 − blind-direction share). */
  visibleFrac: number;
}

export interface LensReport {
  rungs: LensRung[];
  /** log₂(V) — the zero-knowledge rung every ladder starts from. */
  uniformBits: number;
}

/** Run the lens over the model's most recent forward pass. */
export function logitLens(model: MiniGPT): LensReport {
  const t = model.lastTensors;
  if (!t) throw new Error('logitLens: call forward() first');
  const [T, D] = t.blockOut.shape;
  const V = model.cfg.vocabSize;
  const off = (T - 1) * D;
  const lOff = (T - 1) * V;

  // Shared readout: final LayerNorm + unembed on a single residual row —
  // p = softmax( Wᵀ·(γ ⊙ x̂ + β) + b ) with x̂ = (h − μ)/√(σ² + ε).
  // A scalar re-implementation of lnFinal + head so it can be applied to any
  // D-vector (raw rung or J-transported), not just the model's own blockOut.
  const gam = model.lnFinal.gamma.data;
  const bet = model.lnFinal.beta.data;
  const W = model.head.weight.data; // [D, V]
  const bias = model.head.bias.data;
  const readout = (h: Float64Array): Float64Array => {
    let mu = 0;
    for (let d = 0; d < D; d++) mu += h[d];
    mu /= D;
    let va = 0;
    for (let d = 0; d < D; d++) va += (h[d] - mu) ** 2;
    va /= D;
    const inv = 1 / Math.sqrt(va + 1e-5);
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) logits[v] = bias[v];
    for (let d = 0; d < D; d++) {
      const normed = (h[d] - mu) * inv * gam[d] + bet[d];
      for (let v = 0; v < V; v++) logits[v] += normed * W[d * V + v];
    }
    return softmaxVec(logits);
  };

  // Transport Jacobians J[d][e] = ∂ blockOut[last,d] / ∂ rung[last,e]:
  // D sweeps, each seeding one output dim of the final residual. Reverse mode
  // yields one *row* of J per sweep — after seeding output d, every upstream
  // grad holds ∂blockOut[d]/∂(that value), and slicing the last position's D
  // entries reads the row off both rungs from the same sweep.
  const Jemb: Float64Array[] = [];
  const Jattn: Float64Array[] = [];
  for (let d = 0; d < D; d++) {
    backward(t.blockOut, off + d);
    Jemb.push(t.embedded.grad.slice(off, off + D));
    Jattn.push(t.attnResid.grad.slice(off, off + D));
  }

  // Output-sensitivity Jacobians Jl[v][e] = ∂ logits[last,v] / ∂ rung[last,e]:
  // V sweeps, each seeding one logit. These go all the way through lnFinal +
  // unembed, so their row space is exactly the set of residual directions the
  // output can perceive (used by visibleFraction, not by the decoders).
  const JlEmb: Float64Array[] = [];
  const JlAttn: Float64Array[] = [];
  const JlOut: Float64Array[] = [];
  for (let v = 0; v < V; v++) {
    backward(t.logits, lOff + v);
    JlEmb.push(t.embedded.grad.slice(off, off + D));
    JlAttn.push(t.attnResid.grad.slice(off, off + D));
    JlOut.push(t.blockOut.grad.slice(off, off + D));
  }

  // (J·h)[d] = Σ_e J[d][e]·h[e]: the rung pushed through the *linearisation* of
  // the remaining layers at the actual forward point — first-order transport
  // into the final basis, exact only if what remains were linear.
  const transported = (J: Float64Array[], h: Float64Array): Float64Array => {
    const out = new Float64Array(D);
    for (let d = 0; d < D; d++) {
      let s = 0;
      for (let e = 0; e < D; e++) s += J[d][e] * h[e];
      out[d] = s;
    }
    return out;
  };

  const hEmb = t.embedded.data.slice(off, off + D);
  const hAttn = t.attnResid.data.slice(off, off + D);
  const hOut = t.blockOut.data.slice(off, off + D);
  const finalProbs = readout(hOut); // identical to the model's own output

  const rungs: LensRung[] = [
    {
      key: 'emb',
      label: 'token + position',
      resid: hEmb,
      naiveProbs: readout(hEmb),
      jProbs: readout(transported(Jemb, hEmb)),
      visibleFrac: visibleFraction(JlEmb, hEmb)
    },
    {
      key: 'attn',
      label: '+ attention',
      resid: hAttn,
      naiveProbs: readout(hAttn),
      jProbs: readout(transported(Jattn, hAttn)),
      visibleFrac: visibleFraction(JlAttn, hAttn)
    },
    {
      key: 'ffn',
      label: '+ feed-forward',
      resid: hOut,
      naiveProbs: finalProbs,
      jProbs: finalProbs, // J = identity at the top of the ladder
      visibleFrac: visibleFraction(JlOut, hOut)
    }
  ];

  return { rungs, uniformBits: Math.log2(V) };
}

/**
 * ‖P h‖² / ‖h‖² where P projects onto the row space of the (vocab-centered)
 * output Jacobian. Centering over v first: softmax only sees logit
 * *differences*, so a direction that shifts every logit equally is still blind.
 */
function visibleFraction(Jl: Float64Array[], h: Float64Array): number {
  const V = Jl.length;
  const D = h.length;

  const mean = new Float64Array(D);
  for (let v = 0; v < V; v++) for (let d = 0; d < D; d++) mean[d] += Jl[v][d] / V;

  // Gram–Schmidt over the centered rows → orthonormal basis of the visible
  // space: subtract each accepted basis vector's component, keep what survives.
  // Rows nearly in the span of earlier ones shrink to ~0 and are dropped by the
  // relative tolerance (1e-9 × largest entry), so the basis size = numerical
  // rank of the centered Jacobian (≤ min(V−1, D)).
  const basis: Float64Array[] = [];
  let scale = 0;
  for (let v = 0; v < V; v++)
    for (let d = 0; d < D; d++) scale = Math.max(scale, Math.abs(Jl[v][d] - mean[d]));
  for (let v = 0; v < V; v++) {
    const r = new Float64Array(D);
    for (let d = 0; d < D; d++) r[d] = Jl[v][d] - mean[d];
    for (const b of basis) {
      let dot = 0;
      for (let d = 0; d < D; d++) dot += r[d] * b[d];
      for (let d = 0; d < D; d++) r[d] -= dot * b[d];
    }
    let n = 0;
    for (let d = 0; d < D; d++) n += r[d] * r[d];
    n = Math.sqrt(n);
    if (n > 1e-9 * (scale || 1)) {
      for (let d = 0; d < D; d++) r[d] /= n;
      basis.push(r);
    }
  }

  // With {b} orthonormal, ‖P h‖² = Σ_b ⟨h,b⟩² (Parseval); ratio to ‖h‖² is the
  // energy fraction the output can see. min(1,·) absorbs float round-off.
  let h2 = 0;
  for (let d = 0; d < D; d++) h2 += h[d] * h[d];
  if (h2 === 0) return 0;
  let p2 = 0;
  for (const b of basis) {
    let dot = 0;
    for (let d = 0; d < D; d++) dot += h[d] * b[d];
    p2 += dot * dot;
  }
  return Math.min(1, p2 / h2);
}

/**
 * Bits view of a probability: the code length of `id` under `probs`.
 * Shannon/arithmetic-coding cost −log₂ p; the 1e-12 floor caps a zero-prob
 * token at ~40 bits instead of ∞.
 */
export function bitsOf(probs: Float64Array, id: number): number {
  return -Math.log2(probs[id] + 1e-12);
}
