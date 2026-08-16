import { describe, expect, it } from 'bun:test';
import { trainTrace } from './trainTrace';
import { DATASETS } from './datasets';

const ds = DATASETS['color-arithmetic'];
const OPTS = { steps: 40, seed: 3 };

const meanAbsWeights = (w: Float64Array) => {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += Math.abs(w[i]);
  return s / w.length;
};

describe('trainTrace options', () => {
  it('absent opts and oversized evalSample are bit-identical', () => {
    const base = trainTrace(ds, OPTS);
    const sampled = trainTrace(ds, { ...OPTS, evalSample: ds.trainData.length + 5 });
    expect(sampled.steps.map((s) => s.loss)).toEqual(base.steps.map((s) => s.loss));
    expect([...sampled.weights[base.weights.length - 1]]).toEqual([...base.weights[base.weights.length - 1]]);
  });

  it('a real subsample changes only the reported curve, not the training path', () => {
    const base = trainTrace(ds, OPTS);
    const sampled = trainTrace(ds, { ...OPTS, evalSample: 2 });
    // Same weights (the update stream is untouched)…
    expect([...sampled.weights[base.weights.length - 1]]).toEqual([...base.weights[base.weights.length - 1]]);
    // …but losses evaluated on 2 of 6 sequences generally differ.
    const same = sampled.steps.every((s, i) => s.loss === base.steps[i].loss);
    expect(same).toBe(false);
  });

  it('l1Lambda shrinks mean |w| at equal seed/steps', () => {
    const plain = trainTrace(ds, OPTS);
    const sparse = trainTrace(ds, { ...OPTS, l1Lambda: 0.01 });
    const last = plain.weights.length - 1;
    expect(meanAbsWeights(sparse.weights[last])).toBeLessThan(meanAbsWeights(plain.weights[last]));
  });

  it('l1Lambda: 0 is a no-op', () => {
    const plain = trainTrace(ds, OPTS);
    const zero = trainTrace(ds, { ...OPTS, l1Lambda: 0 });
    expect([...zero.weights[zero.weights.length - 1]]).toEqual([...plain.weights[plain.weights.length - 1]]);
  });
});
