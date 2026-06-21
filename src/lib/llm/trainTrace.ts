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
import { MiniGPT, Adam, type ModelConfig, type ForwardViz } from './model';
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
}

export function trainTrace(ds: Dataset, opts: TrainTraceOptions = {}): LlmStep[] {
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

  /** Mean cross-entropy across the whole training set (no gradient). */
  const meanLoss = (): number => {
    let sum = 0;
    for (const { input, targets } of train) {
      const logits = model.forward(input);
      sum += crossEntropyLoss(logits, targets).data[0];
    }
    return sum / train.length;
  };

  const out: LlmStep[] = [];

  for (let i = 0; i < steps; i++) {
    const loss = meanLoss();

    // Probe forward LAST so `model.viz` reflects the probe, not the loss eval.
    model.forward(probeInput);
    const viz = model.viz!;

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
    const ex = train[randInt(rng, train.length)];
    optimizer.zeroGrad();
    const logits = model.forward(ex.input);
    const loss2 = crossEntropyLoss(logits, ex.targets);
    backward(loss2);
    optimizer.step();
  }

  return out;
}
