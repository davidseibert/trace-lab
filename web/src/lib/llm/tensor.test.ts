import { describe, expect, it } from 'bun:test';
import {
  Tensor,
  backward,
  crossEntropyLoss,
  maskedSoftmax,
  reshape,
  softCrossEntropyLoss,
  softmaxRows
} from './tensor';

const t = (data: number[], shape: number[]) => new Tensor(new Float64Array(data), shape);

describe('softmaxRows', () => {
  it('rows sum to 1 and match maskedSoftmax on the last (unmasked) row', () => {
    const a = t([1, 2, 3, 0.5, -1, 2, 0, 0, 0], [3, 3]);
    const full = softmaxRows(a);
    for (let r = 0; r < 3; r++) {
      let s = 0;
      for (let c = 0; c < 3; c++) s += full.data[r * 3 + c];
      expect(s).toBeCloseTo(1, 12);
    }
    // maskedSoftmax's final row sees every column — identical to the full softmax.
    const causal = maskedSoftmax(t([1, 2, 3, 0.5, -1, 2, 0, 0, 0], [3, 3]), 3);
    for (let c = 0; c < 3; c++) expect(causal.data[6 + c]).toBeCloseTo(full.data[6 + c], 12);
  });

  it('backward matches a numerical gradient', () => {
    const mk = () => t([0.3, -1.2, 0.7, 2.0, 0.1, -0.5], [2, 3]);
    const a = mk();
    const out = softmaxRows(a);
    // Scalar objective: Σ wᵢ·pᵢ with fixed weights, via manual seeding.
    const w = [0.2, -0.4, 0.9, 0.1, 0.5, -0.3];
    out.grad.set(w);
    a.grad.fill(0);
    out._backward();
    const eps = 1e-6;
    for (let i = 0; i < 6; i++) {
      const ap = mk();
      ap.data[i] += eps;
      const am = mk();
      am.data[i] -= eps;
      const f = (x: Tensor) => {
        const p = softmaxRows(x);
        let s = 0;
        for (let j = 0; j < 6; j++) s += w[j] * p.data[j];
        return s;
      };
      expect(a.grad[i]).toBeCloseTo((f(ap) - f(am)) / (2 * eps), 6);
    }
  });
});

describe('softCrossEntropyLoss', () => {
  it('equals crossEntropyLoss when q is one-hot', () => {
    const logits = () => t([0.2, -1.1, 0.8, 1.4, 0.3, -0.2, 0.9, 0.0], [2, 4]);
    const hard = crossEntropyLoss(logits(), [2, 0]);
    const q = new Float64Array(8);
    q[2] = 1;
    q[4] = 1;
    const soft = softCrossEntropyLoss(logits(), q);
    expect(soft.data[0]).toBeCloseTo(hard.data[0], 10);
    // Gradients agree too.
    const lh = logits();
    backward(crossEntropyLoss(lh, [2, 0]));
    const ls = logits();
    backward(softCrossEntropyLoss(ls, q));
    for (let i = 0; i < 8; i++) expect(ls.grad[i]).toBeCloseTo(lh.grad[i], 10);
  });

  it('backward matches a numerical gradient for a genuinely soft q', () => {
    const q = new Float64Array([0.5, 0.25, 0.25]);
    const mk = () => t([0.4, -0.6, 1.1], [1, 3]);
    const l = mk();
    backward(softCrossEntropyLoss(l, q));
    const eps = 1e-6;
    for (let i = 0; i < 3; i++) {
      const lp = mk();
      lp.data[i] += eps;
      const lm = mk();
      lm.data[i] -= eps;
      const num = (softCrossEntropyLoss(lp, q).data[0] - softCrossEntropyLoss(lm, q).data[0]) / (2 * eps);
      expect(l.grad[i]).toBeCloseTo(num, 6);
    }
  });
});

describe('reshape', () => {
  it('round-trips data and passes gradients through', () => {
    const a = t([1, 2, 3, 4, 5, 6], [3, 2]);
    const r = reshape(a, [1, 6]);
    expect([...r.data]).toEqual([1, 2, 3, 4, 5, 6]);
    expect(r.shape).toEqual([1, 6]);
    r.grad.set([9, 8, 7, 6, 5, 4]);
    a.grad.fill(0);
    r._backward();
    expect([...a.grad]).toEqual([9, 8, 7, 6, 5, 4]);
    expect(() => reshape(a, [4, 2])).toThrow();
  });
});
