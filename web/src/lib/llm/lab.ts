/**
 * lab — the replay-and-surgery workbench the UI drives.
 *
 * A `TrainRun` gives per-step weight snapshots; the lab turns (stepIndex,
 * facts) into a runnable model — the base model, or a grown one with every
 * fact grafted in — and answers three questions about it:
 *
 *   forward()      what does this model do on this input? (panels)
 *   lens()         what does its residual ladder decode to, rung by rung?
 *   interference() what did the implants do to *everything else*, in Δbits?
 *
 * Implants are re-applied on top of whichever step is selected, so scrubbing
 * the timeline replays the same surgery against earlier (dumber) weights —
 * early-training keys are undifferentiated, and the interference table shows
 * exactly what that costs.
 *
 * One-entry memo: playback hammers the same (step, facts) pair many times per
 * scrub, and implantModel() re-runs key extraction each build.
 */

import { MiniGPT, loadParams, type ForwardViz } from './model';
import { mulberry32 } from './rng';
import { softmaxVec } from './tensor';
import { logitLens, type LensReport } from './lens';
import { implantModel, type FactSpec, type ImplantUnit } from './implant';
import type { TrainRun } from './trainTrace';

export interface EvalCase {
  label: string;
  inputIds: number[];
  /**
   * Which transitions to charge for: position `pos`'s distribution must pay for
   * token `id`. A full sequence lists every transition — its bits are then the
   * sequence's literal description length `L(seq|M)`. A fact lists only its
   * final transition.
   */
  targets: { pos: number; id: number }[];
  /** True when this case IS one of the implanted facts (rendered highlighted). */
  isFact?: boolean;
}

export interface InterferenceRow {
  label: string;
  /** Σ −log₂ p over the case's transitions, before / after the implants. */
  before: number;
  after: number;
  isFact: boolean;
  /** A fact's prompt is a prefix of this input — the overwrite hitting its own
   * ground truth, not collateral interference. */
  touched: boolean;
}

export interface Lab {
  forward(step: number, tokenIds: number[], facts: FactSpec[]): { viz: ForwardViz; units: ImplantUnit[] };
  lens(step: number, tokenIds: number[], facts: FactSpec[]): LensReport;
  interference(step: number, facts: FactSpec[], cases: EvalCase[]): InterferenceRow[];
}

export function makeLab(run: TrainRun, refInputs: number[][]): Lab {
  const baseModel = new MiniGPT(run.cfg, mulberry32(1));
  const scratch = new MiniGPT(run.cfg, mulberry32(1));

  let memoKey = '';
  let memoModel: MiniGPT = baseModel;
  let memoUnits: ImplantUnit[] = [];

  function modelAt(step: number, facts: FactSpec[]): { model: MiniGPT; units: ImplantUnit[] } {
    const key = `${step}§${JSON.stringify(facts)}`;
    if (key !== memoKey) {
      if (facts.length === 0) {
        loadParams(baseModel, run.weights[step]);
        memoModel = baseModel;
        memoUnits = [];
      } else {
        const r = implantModel(run.cfg, run.weights[step], facts, refInputs);
        memoModel = r.model;
        memoUnits = r.units;
      }
      memoKey = key;
    }
    return { model: memoModel, units: memoUnits };
  }

  const V = run.cfg.vocabSize;
  const caseBits = (model: MiniGPT, c: EvalCase): number => {
    model.forward(c.inputIds);
    const logits = model.viz!.logits;
    let bits = 0;
    for (const { pos, id } of c.targets) {
      const p = softmaxVec(logits.data.subarray(pos * V, (pos + 1) * V));
      bits += -Math.log2(p[id] + 1e-12);
    }
    return bits;
  };

  const isPrefix = (prefix: number[], seq: number[]): boolean =>
    prefix.length <= seq.length && prefix.every((id, i) => seq[i] === id);

  return {
    forward(step, tokenIds, facts) {
      const { model, units } = modelAt(step, facts);
      model.forward(tokenIds);
      return { viz: model.viz!, units };
    },

    lens(step, tokenIds, facts) {
      const { model } = modelAt(step, facts);
      model.forward(tokenIds);
      return logitLens(model);
    },

    interference(step, facts, cases) {
      loadParams(scratch, run.weights[step]);
      const { model } = modelAt(step, facts);
      return cases.map((c) => ({
        label: c.label,
        before: caseBits(scratch, c),
        after: caseBits(model, c),
        isFact: c.isFact ?? false,
        touched: !c.isFact && facts.some((f) => isPrefix(f.promptIds, c.inputIds))
      }));
    }
  };
}
