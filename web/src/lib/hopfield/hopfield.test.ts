import { describe, expect, it } from 'bun:test';
import { mulberry32 } from '../llm/rng';
import { GLYPH_DIM, glyphVector, validGlyphChars } from './glyphs';
import {
  betaSweep,
  capacityExperiment,
  classifyRegime,
  corrupt,
  hebbianWeights,
  logSpacedBetas,
  modernEnergy,
  randomPatterns,
  runClassical,
  runModern,
  separationStats,
  zipTraces
} from './hopfield';
import type { TensorSnap } from '../llm/tensor';

const BETA_STAR = 1 / Math.sqrt(GLYPH_DIM);

function glyphMatrix(chars: string): TensorSnap {
  const list = validGlyphChars(chars);
  const data = new Float64Array(list.length * GLYPH_DIM);
  list.forEach((ch, i) => data.set(glyphVector(ch), i * GLYPH_DIM));
  return { data, shape: [list.length, GLYPH_DIM] };
}

describe('modern network', () => {
  const X = glyphMatrix('ABCX');

  it('weights are a distribution at every step', () => {
    const { modern } = corrupt(glyphVector('A'), mulberry32(1), { flipFrac: 0.15 });
    const trace = runModern(X, modern, 4 * BETA_STAR);
    for (const s of trace.steps) {
      const sum = s.weights.reduce((a, v) => a + v, 0);
      expect(sum).toBeCloseTo(1, 9);
      for (const w of s.weights) expect(w).toBeGreaterThanOrEqual(0);
    }
  });

  it('energy is non-negative and non-increasing along the trace', () => {
    const { modern } = corrupt(glyphVector('B'), mulberry32(2), { flipFrac: 0.2 });
    const trace = runModern(X, modern, 2 * BETA_STAR);
    for (let t = 0; t < trace.steps.length; t++) {
      expect(trace.steps[t].energyNats).toBeGreaterThanOrEqual(0);
      if (t > 0) expect(trace.steps[t].energyNats).toBeLessThanOrEqual(trace.steps[t - 1].energyNats + 1e-9);
    }
  });

  it('one-step convergence: mild corruption at high β is ε-done by iter 2', () => {
    const { modern } = corrupt(glyphVector('A'), mulberry32(3), { flipFrac: 0.1 });
    const trace = runModern(X, modern, 4 * BETA_STAR, { tol: 1e-4 });
    expect(trace.converged).toBe(true);
    const done = trace.steps.findIndex((s, i) => i > 0 && s.deltaNorm < 1e-4);
    expect(done).toBeLessThanOrEqual(2);
    expect(trace.retrieved).toBe(0); // 'A'
    expect(trace.regime).toBe('retrieval');
  });

  it('a stored pattern is (numerically) a fixed point at high β', () => {
    const trace = runModern(X, glyphVector('C'), 10 * BETA_STAR, { tol: 1e-6 });
    expect(trace.converged).toBe(true);
    expect(trace.retrieved).toBe(2); // 'C'
  });

  it('low β lands in the global regime', () => {
    const { modern } = corrupt(glyphVector('A'), mulberry32(4), { flipFrac: 0.15 });
    const trace = runModern(X, modern, 0.02 * BETA_STAR);
    expect(trace.regime).toBe('global');
  });

  it('determinism: same seed ⇒ identical trace', () => {
    const a = corrupt(glyphVector('X'), mulberry32(9), { flipFrac: 0.25 }).modern;
    const b = corrupt(glyphVector('X'), mulberry32(9), { flipFrac: 0.25 }).modern;
    expect([...a]).toEqual([...b]);
    const ta = runModern(X, a, BETA_STAR);
    const tb = runModern(X, b, BETA_STAR);
    expect(ta.steps.length).toBe(tb.steps.length);
    ta.steps.forEach((s, i) => expect([...s.xi]).toEqual([...tb.steps[i].xi]));
  });
});

describe('classical network', () => {
  const X = glyphMatrix('ATX');

  it('retrieves a mildly corrupted pattern with few patterns stored', () => {
    const { classical } = corrupt(glyphVector('T'), mulberry32(5), { flipFrac: 0.1 });
    const trace = runClassical(X, classical);
    expect(trace.converged).toBe(true);
    expect(trace.retrieved).toBe(1); // 'T'
  });

  it('energy is non-increasing across sweeps', () => {
    const { classical } = corrupt(glyphVector('A'), mulberry32(6), { flipFrac: 0.3 });
    const trace = runClassical(X, classical);
    for (let t = 1; t < trace.steps.length; t++)
      expect(trace.steps[t].energy).toBeLessThanOrEqual(trace.steps[t - 1].energy + 1e-9);
  });

  it('hebbian weights have zero diagonal and are symmetric', () => {
    const W = hebbianWeights(X);
    const d = GLYPH_DIM;
    for (let a = 0; a < d; a++) {
      expect(W[a * d + a]).toBe(0);
      for (let b = 0; b < d; b++) expect(W[a * d + b]).toBeCloseTo(W[b * d + a], 12);
    }
  });
});

describe('regimes and helpers', () => {
  it('classifyRegime boundary cases', () => {
    expect(classifyRegime(new Float64Array([1, 0, 0, 0]), 4)).toBe('retrieval');
    expect(classifyRegime(new Float64Array([0.25, 0.25, 0.25, 0.25]), 4)).toBe('global');
    expect(classifyRegime(new Float64Array([0.48, 0.48, 0.02, 0.02]), 4)).toBe('metastable');
  });

  it('separationStats finds the least separated pair', () => {
    const X = glyphMatrix('ABH'); // A and H share most strokes
    const { delta, minPair } = separationStats(X);
    expect(delta).toBeLessThan(GLYPH_DIM);
    expect(minPair[0]).not.toBe(minPair[1]);
  });

  it('β sweep is monotone in spirit: max weight grows with β', () => {
    const X = glyphMatrix('ABCX');
    const { modern } = corrupt(glyphVector('A'), mulberry32(7), { flipFrac: 0.15 });
    const pts = betaSweep(X, modern, logSpacedBetas(1 / Math.sqrt(GLYPH_DIM)));
    expect(pts[0].maxWeight).toBeLessThan(pts[pts.length - 1].maxWeight);
    expect(pts[0].regime).toBe('global');
    expect(pts[pts.length - 1].regime).toBe('retrieval');
  });

  it('corrupt: mask zeroes the modern bottom half, classical stays ±1', () => {
    const { modern, classical } = corrupt(glyphVector('A'), mulberry32(8), { flipFrac: 0, mask: 'bottom' });
    const d = GLYPH_DIM;
    for (let j = Math.floor(d / 2); j < d; j++) {
      expect(modern[j]).toBe(0);
      expect(Math.abs(classical[j])).toBe(1);
    }
    for (let j = 0; j < Math.floor(d / 2); j++) expect(modern[j]).toBe(classical[j]);
  });

  it('zipTraces pads the shorter trace by holding its last step', () => {
    const X = glyphMatrix('AB');
    const { modern, classical } = corrupt(glyphVector('A'), mulberry32(10), { flipFrac: 0.2 });
    const m = runModern(X, modern, 1);
    const c = runClassical(X, classical);
    const zipped = zipTraces(m, c);
    expect(zipped.length).toBe(Math.max(m.steps.length, c.steps.length));
    expect(zipped[zipped.length - 1].modern).toBe(m.steps[m.steps.length - 1]);
    expect(zipped[zipped.length - 1].classical).toBe(c.steps[c.steps.length - 1]);
  });
});

describe('capacity', () => {
  it('modern beats classical at high N on random ±1 patterns', () => {
    const d = 64;
    const pts = capacityExperiment({
      Ns: [4, 32],
      trials: 10,
      noiseFrac: 0.15,
      beta: 1 / Math.sqrt(d),
      seed: 1,
      draw: (N, rng) => randomPatterns(N, d, rng)
    });
    const at32 = pts.find((p) => p.N === 32)!;
    // Classical capacity ~0.138·d ≈ 9 patterns at d=64: 32 must be far gone.
    expect(at32.classical).toBeLessThan(0.5);
    expect(at32.modern).toBeGreaterThan(at32.classical);
    const at4 = pts.find((p) => p.N === 4)!;
    expect(at4.modern).toBeGreaterThanOrEqual(0.9);
  });

  it('modernEnergy stays finite at extreme β', () => {
    const X = glyphMatrix('AB');
    const xi = glyphVector('A');
    expect(Number.isFinite(modernEnergy(X, xi, 1e-4))).toBe(true);
    expect(Number.isFinite(modernEnergy(X, xi, 1e3))).toBe(true);
  });
});
