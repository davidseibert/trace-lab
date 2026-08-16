/**
 * trainTrace — the transformer lens's answer to the MDL lens's `trace()`.
 *
 * It runs the whole training run to completion up front and emits an immutable
 * `LlmStep[]`. The UI is then a pure function of `steps[index]`, so the existing
 * `Player` gives us play / pause / step / scrub — including scrubbing BACKWARD
 * through training, which a live training loop can't offer.
 *
 * Two things make this honest and reproducible:
 *   1. Weight init AND example sampling draw from one seeded RNG, so a given
 *      (dataset, seed, steps) always yields the identical trace.
 *   2. Each step reports the MEAN loss over the whole training set (a clean
 *      learning curve, not the noisy per-sample loss) and visualises a FIXED
 *      probe input, so the panels show one sentence's internals sharpening.
 *
 * Step i holds the model state BEFORE the i-th update — mirroring the MDL engine,
 * where `steps[i]` is the state and `chosen` is what happens next. The final
 * step has `chosen === false`.
 */

import { mulberry32, randInt, type Rng } from './rng';
import {
  MiniGPT,
  Adam,
  flattenParams,
  loadParams,
  type ModelConfig,
  type ForwardViz
} from './model';
import { crossEntropyLoss, backward } from './tensor';
import type { Dataset } from './datasets';

export interface LlmStep {
  /** Training step index (0 = randomly-initialised, before any update). */
  index: number;
  /** Mean cross-entropy over the whole training set at this state, in nats. */
  loss: number;
  /** Forward pass of the fixed probe input — what the panels render. */
  viz: ForwardViz;
  /** Argmax next-token id for the probe. */
  predId: number;
  /** Human label of the predicted token. */
  predToken: string;
  /** Probability mass on the predicted token (peakedness of the guess). */
  confidence: number;
  /** Probability the model assigns to the *correct* probe continuation. */
  targetProb: number;
  /** Whether a further training update follows this step. */
  chosen: boolean;
  /** Narration for the transport bar. */
  note: string;
}

export interface TrainTraceOptions {
  /** Number of snapshots (training updates run = steps - 1). */
  steps?: number;
  /** RNG seed; same seed ⇒ identical trace. */
  seed?: number;
  /**
   * L1 penalty coefficient on ALL weights (embeddings included, mirroring the
   * classic sparse-circuits recipe): after backward(), each param gets
   * grad[i] += l1Lambda·sign(data[i]) before the Adam step. Injected as a
   * hand-computed gradient because backward() is single-root and destructive —
   * a second loss term can't be backward'd separately. 0/undefined = off.
   */
  l1Lambda?: number;
  /**
   * Cap the per-step meanLoss sweep to a fixed subsample of the training set
   * (chosen once, up front, from its own rng so the training sample stream is
   * unaffected). Needed for big corpora — the full sweep is |train| forwards
   * per step. undefined, or ≥ trainData.length, = the full set, bit-identical
   * to the option being absent.
   */
  evalSample?: number;
}

/**
 * The full precomputed run: the visible step trace plus everything needed to
 * replay an *arbitrary* input against the model as it stood at any step (the
 * flat parameter snapshot per step). That's what powers the live query box.
 */
export interface TrainRun {
  steps: LlmStep[];
  cfg: ModelConfig;
  vocab: string[];
  /** weights[i] = flat parameters at step i (before the i-th update). */
  weights: Float64Array[];
}

export function trainTrace(ds: Dataset, opts: TrainTraceOptions = {}): TrainRun {
  const steps = opts.steps ?? 300;
  const seed = opts.seed ?? 1;
  const rng: Rng = mulberry32(seed);

  const tok2id = new Map<string, number>();
  ds.vocab.forEach((w, i) => tok2id.set(w, i));
  const ids = (words: string[]) => words.map((w) => tok2id.get(w)!);

  const cfg: ModelConfig = {
    vocabSize: ds.vocab.length,
    embDim: ds.embDim,
    ctxLen: ds.ctxLen,
    nHeads: ds.nHeads,
    ffnHid: ds.ffnHid
  };

  const model = new MiniGPT(cfg, rng);
  const optimizer = new Adam(model.params(), ds.lr);

  // Pre-tokenise: each training sequence split into (input, targets) by shift.
  const train = ds.trainData.map((seq) => {
    const t = ids(seq);
    return { input: t.slice(0, -1), targets: t.slice(1) };
  });

  const probeInput = ids(ds.probe);
  const probeTargetId = tok2id.get(ds.probeTarget)!;

  // The loss-curve eval set: the whole training set, or a fixed seeded
  // subsample of it. The subsample draws from its OWN rng (not `rng`), so
  // turning the option on never shifts weight init or example sampling.
  let evalSet = train;
  if (opts.evalSample !== undefined && train.length > opts.evalSample) {
    const sub = mulberry32(seed ^ 0x9e3779b9);
    const idx = [...train.keys()];
    // Partial Fisher–Yates: the first evalSample entries are a uniform sample.
    for (let k = 0; k < opts.evalSample; k++) {
      const r = k + randInt(sub, idx.length - k);
      [idx[k], idx[r]] = [idx[r], idx[k]];
    }
    evalSet = idx.slice(0, opts.evalSample).map((i) => train[i]);
  }

  /** Mean cross-entropy across the eval set (no gradient). */
  const meanLoss = (): number => {
    let sum = 0;
    for (const { input, targets } of evalSet) {
      const logits = model.forward(input);
      sum += crossEntropyLoss(logits, targets).data[0];
    }
    return sum / evalSet.length;
  };

  const out: LlmStep[] = [];
  const weights: Float64Array[] = [];

  for (let i = 0; i < steps; i++) {
    const loss = meanLoss();

    // Snapshot the weights AT this state, so the query box can replay any input
    // against the model exactly as it stood at step i.
    weights.push(flattenParams(model));

    // Probe forward LAST so `model.viz` reflects the probe, not the loss eval.
    model.forward(probeInput);
    const viz = model.viz!;

    // Greedy readout: viz.probs is already the softmaxed next-token
    // distribution at the probe's last position ([vocabSize]), so a linear
    // argmax scan gives the model's top-1 guess.
    let predId = 0;
    for (let v = 1; v < viz.probs.length; v++) if (viz.probs[v] > viz.probs[predId]) predId = v;
    const confidence = viz.probs[predId];
    const targetProb = viz.probs[probeTargetId];
    const predToken = ds.vocab[predId];
    const isLast = i === steps - 1;

    out.push({
      index: i,
      loss,
      viz,
      predId,
      predToken,
      confidence,
      targetProb,
      chosen: !isLast,
      note: isLast
        ? `Trained ${steps - 1} updates — final mean loss ${loss.toFixed(3)}. ` +
          `Probe predicts “${predToken}” (${(confidence * 100).toFixed(0)}%).`
        : `Step ${i}: mean loss ${loss.toFixed(3)} — probe leans “${predToken}” ` +
          `(${(confidence * 100).toFixed(0)}%), correct answer “${ds.probeTarget}” at ` +
          `${(targetProb * 100).toFixed(0)}%.`
    });

    if (isLast) break;

    // One training update: sample an example, forward, backward, Adam step.
    // Batch size 1 (pure SGD-style sampling) — the noisy per-sample gradient is
    // fine because the reported curve is meanLoss(), not this sample's loss.
    const ex = train[randInt(rng, train.length)];
    optimizer.zeroGrad();
    const logits = model.forward(ex.input);
    const loss2 = crossEntropyLoss(logits, ex.targets);
    backward(loss2);
    if (opts.l1Lambda) {
      // d|w|/dw = sign(w) (0 at 0, so dead weights stay dead) — the L1 pull
      // toward zero that makes the circuit-view threshold meaningful.
      for (const p of model.params()) {
        for (let j = 0; j < p.data.length; j++) p.grad[j] += opts.l1Lambda * Math.sign(p.data[j]);
      }
    }
    optimizer.step();
  }

  return { steps: out, cfg, vocab: ds.vocab, weights };
}

/**
 * Build a reusable forward function over a finished run: given a step index and
 * any token-id sequence, load that step's weights and return the forward-pass
 * snapshot. One model instance is reused across calls (its random init is
 * irrelevant — every parameter is overwritten by loadParams).
 */
export function makeForward(run: TrainRun): (stepIndex: number, tokenIds: number[]) => ForwardViz {
  const model = new MiniGPT(run.cfg, mulberry32(1));
  return (stepIndex, tokenIds) => {
    loadParams(model, run.weights[stepIndex]);
    model.forward(tokenIds);
    return model.viz!;
  };
}
