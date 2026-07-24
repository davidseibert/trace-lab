/**
 * implant — write a fact into the mini-GPT's FFN in closed form, no training.
 *
 * The Hebbian-memory reading of a transformer MLP: `fc1` columns are stored
 * *keys*, `fc2` rows are stored *values*, and ReLU gates recall. So implanting
 * "prompt → target" is literally one outer-product write:
 *
 *   key   k  = the FFN's actual input (ln2 of the residual) at the prompt's
 *              last position — the address the network itself would present
 *   value v  = the unembed column of the target token, centered over the vocab
 *              (push the target *relative* to everyone, not all logits at once)
 *   slot     = one brand-new hidden unit:  fc1 col = k̂,  bias = −θ·(k̂·x),
 *              fc2 row = (strength / activation) · v̂
 *
 * The bias makes the unit a *gated* memory: it fires only where the incoming
 * vector aligns with the key at least θ of as well as the fact's own prompt
 * does. The unit is appended (ffnHid grows), so the new memory slot is visible
 * as an extra column in the FFN activation grid.
 *
 * WHITEN is the interesting toggle. Raw keys all share the corpus-mean
 * component (anisotropy), so raw-key implants fire on *everything* — that
 * shared direction carries zero discriminative bits but dominates the dot
 * product. Subtracting the mean FFN-input μ (estimated over the reference
 * prompts) before storing the key is exactly the associative-memory capacity
 * condition: whitened keys interfere far less. The interference table makes
 * the difference measurable in bits.
 */

import { MiniGPT, loadParams, type ModelConfig } from './model';
import { mulberry32 } from './rng';

export interface FactSpec {
  /** Stable identity for the UI list. */
  key: string;
  promptIds: number[];
  targetId: number;
  /**
   * Target-logit boost at the fact's own position (calibrated through the real
   * lnFinal, so it means the same thing on every dataset): the write multiplies
   * the target's odds by ≈ e^strength.
   */
  strength: number;
  /** Subtract the shared (mean) component from the key before storing. */
  whiten: boolean;
  /** Fires when alignment exceeds this fraction of the fact's own alignment. */
  threshold: number;
}

export interface ImplantUnit {
  factKey: string;
  /** Column index of the grafted unit in the grown FFN. */
  unit: number;
  /** k̂·x at the fact's own prompt — how addressable the key is. */
  align: number;
  /** ‖k‖ after optional whitening — how much key survives the mean-subtract. */
  keyNorm: number;
}

/**
 * Build a model from `flat` weights with every fact grafted in. Keys are read
 * off the *base* model — safe in a one-block model, because implanted units
 * only alter the FFN output, which never feeds back into ln2's input.
 */
export function implantModel(
  cfg: ModelConfig,
  flat: Float64Array,
  facts: FactSpec[],
  refInputs: number[][]
): { model: MiniGPT; units: ImplantUnit[] } {
  const base = new MiniGPT(cfg, mulberry32(1));
  loadParams(base, flat);

  const D = cfg.embDim;
  const V = cfg.vocabSize;
  const H = cfg.ffnHid;
  const n = facts.length;

  // μ — the shared component of the FFN's input, over every position of every
  // reference prompt. This is the direction whitening removes.
  const mu = new Float64Array(D);
  let count = 0;
  for (const inp of refInputs) {
    base.forward(inp);
    const f = base.block.lastFfnIn!;
    const T = f.shape[0];
    for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) mu[d] += f.data[t * D + d];
    count += T;
  }
  if (count > 0) for (let d = 0; d < D; d++) mu[d] /= count;

  // Grow the FFN by one unit per fact; copy every base weight across.
  const grown = new MiniGPT({ ...cfg, ffnHid: H + n }, mulberry32(1));
  const copy = (dst: Float64Array, src: Float64Array) => dst.set(src);
  copy(grown.tokenEmb.weight.data, base.tokenEmb.weight.data);
  copy(grown.posEmb.weight.data, base.posEmb.weight.data);
  copy(grown.lnFinal.gamma.data, base.lnFinal.gamma.data);
  copy(grown.lnFinal.beta.data, base.lnFinal.beta.data);
  copy(grown.head.weight.data, base.head.weight.data);
  copy(grown.head.bias.data, base.head.bias.data);
  const gb = grown.block;
  const bb = base.block;
  copy(gb.ln1.gamma.data, bb.ln1.gamma.data);
  copy(gb.ln1.beta.data, bb.ln1.beta.data);
  copy(gb.ln2.gamma.data, bb.ln2.gamma.data);
  copy(gb.ln2.beta.data, bb.ln2.beta.data);
  for (const [gp, bp] of [
    [gb.attn.qProj, bb.attn.qProj],
    [gb.attn.kProj, bb.attn.kProj],
    [gb.attn.vProj, bb.attn.vProj],
    [gb.attn.outProj, bb.attn.outProj]
  ] as const) {
    copy(gp.weight.data, bp.weight.data);
    copy(gp.bias.data, bp.bias.data);
  }
  // fc1: [D, H] → [D, H+n], column-wise; new columns start at 0.
  const g1w = gb.ffn.fc1.weight.data;
  const b1w = bb.ffn.fc1.weight.data;
  g1w.fill(0);
  for (let d = 0; d < D; d++) for (let h = 0; h < H; h++) g1w[d * (H + n) + h] = b1w[d * H + h];
  gb.ffn.fc1.bias.data.fill(0);
  gb.ffn.fc1.bias.data.set(bb.ffn.fc1.bias.data.subarray(0, H));
  // fc2: [H, D] → [H+n, D], row-wise; new rows start at 0.
  gb.ffn.fc2.weight.data.fill(0);
  gb.ffn.fc2.weight.data.set(bb.ffn.fc2.weight.data);
  copy(gb.ffn.fc2.bias.data, bb.ffn.fc2.bias.data);

  // Write each fact as key → value into its own unit.
  const headW = base.head.weight.data; // [D, V]
  const units: ImplantUnit[] = facts.map((fact, i) => {
    base.forward(fact.promptIds);
    const f = base.block.lastFfnIn!;
    const T = f.shape[0];
    const x = f.data.slice((T - 1) * D, T * D);

    const k = new Float64Array(D);
    for (let d = 0; d < D; d++) k[d] = x[d] - (fact.whiten ? mu[d] : 0);
    let keyNorm = 0;
    for (let d = 0; d < D; d++) keyNorm += k[d] * k[d];
    keyNorm = Math.sqrt(keyNorm);
    const kn = Math.max(keyNorm, 1e-9);
    let align = 0;
    for (let d = 0; d < D; d++) align += (k[d] / kn) * x[d];
    const a = Math.max(align, 1e-6);
    const act = Math.max(a * (1 - fact.threshold), 1e-6);

    // Value: unembed column of the target, centered over the vocabulary.
    const v = new Float64Array(D);
    let vNorm = 0;
    for (let d = 0; d < D; d++) {
      let rowMean = 0;
      for (let vv = 0; vv < V; vv++) rowMean += headW[d * V + vv];
      rowMean /= V;
      v[d] = headW[d * V + fact.targetId] - rowMean;
      vNorm += v[d] * v[d];
    }
    vNorm = Math.max(Math.sqrt(vNorm), 1e-9);

    const unit = H + i;
    for (let d = 0; d < D; d++) g1w[d * (H + n) + unit] = k[d] / kn;
    gb.ffn.fc1.bias.data[unit] = -fact.threshold * a;
    const fc2 = gb.ffn.fc2.weight.data;
    for (let d = 0; d < D; d++) fc2[unit * D + d] = (1 / act) * (v[d] / vNorm);

    // Calibrate so `strength` = actual target-logit boost at the fact position.
    // lnFinal renormalises whatever we add, so the closed-form magnitude above is
    // only a guess; measure the realised boost and rescale (twice — the response
    // is mildly nonlinear).
    const rowOff = unit * D;
    const targetLogit = (): number => {
      grown.forward(fact.promptIds);
      const lg = grown.viz!.logits;
      return lg.data[(fact.promptIds.length - 1) * V + fact.targetId];
    };
    const saved = fc2.slice(rowOff, rowOff + D);
    fc2.fill(0, rowOff, rowOff + D);
    const l0 = targetLogit();
    fc2.set(saved, rowOff);
    for (let iter = 0; iter < 2; iter++) {
      const delta = targetLogit() - l0;
      const sc = fact.strength / Math.max(delta, 0.05);
      if (Math.abs(sc - 1) < 0.02) break;
      const clamped = Math.min(Math.max(sc, 0.05), 40);
      for (let d = 0; d < D; d++) fc2[rowOff + d] *= clamped;
    }

    return { factKey: fact.key, unit, align, keyNorm };
  });

  return { model: grown, units };
}
