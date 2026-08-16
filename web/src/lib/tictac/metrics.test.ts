import { describe, expect, it } from 'bun:test';
import { trainTrace } from '../llm/trainTrace';
import type { ForwardViz } from '../llm/model';
import { tictacDataset } from './dataset';
import { analyze, INVERSE, isTerminal, TRANSFORMS } from './game';
import {
  buildProbeSuite,
  computeMetrics,
  equivError,
  gameIds,
  groundTruthFeatures,
  makeReplay,
  pca2,
  policyAt,
  sparsityReport,
  unitFeatureCorrelation,
  type ProbePosition
} from './metrics';

describe('probe suite', () => {
  const suite = buildProbeSuite(1);

  it('is deterministic, non-terminal, and agrees with the solver', () => {
    expect(buildProbeSuite(1).map((p) => p.moves)).toEqual(suite.map((p) => p.moves));
    expect(suite.length).toBe(40);
    for (const p of suite) {
      expect(isTerminal(p.board)).toBe(false);
      const a = analyze(p.board);
      expect([...p.optimal].sort()).toEqual([...a.optimal].sort());
    }
  });

  it('features align with FEATURE_NAMES length and range', () => {
    for (const p of suite.slice(0, 5)) {
      const f = groundTruthFeatures(p);
      expect(f.length).toBe(26);
      for (let i = 0; i < 16; i++) expect(f[i]).toBeGreaterThanOrEqual(0);
      expect(Math.abs(f[16])).toBe(1);
    }
  });
});

describe('equivariance metric', () => {
  // A hand-built perfectly equivariant "model": uniform over the position's
  // optimal moves, read from the solver (which is provably equivariant).
  const solverFwd = (_step: number, ids: number[]): ForwardViz => {
    const moves = ids.slice(1).map((t) => t - 1);
    const board = moves.reduce(
      (b, m, i) => ((b[m] = ((i % 2) + 1) as 1 | 2), b),
      new Array(9).fill(0) as (0 | 1 | 2)[]
    );
    const a = analyze(board);
    const probs = new Float64Array(10);
    for (const m of a.optimal) probs[m + 1] = 1 / a.optimal.length;
    return { probs } as unknown as ForwardViz;
  };

  const suite = buildProbeSuite(2, 8);

  it('scores ≈ 0 for the solver policy', () => {
    for (const p of suite) for (let g = 1; g < 8; g++) expect(equivError(solverFwd, 0, p, g)).toBeLessThan(1e-9);
  });

  it('pullback indexing round-trips under INVERSE', () => {
    for (let g = 0; g < 8; g++)
      for (let m = 0; m < 9; m++) expect(TRANSFORMS[INVERSE[g]][TRANSFORMS[g][m]]).toBe(m);
  });

  it('policyAt renormalizes over legal moves', () => {
    const p = suite[0];
    const { pi } = policyAt(solverFwd, 0, p);
    let sum = 0;
    for (const m of p.legal) sum += pi[m];
    expect(sum).toBeCloseTo(1, 9);
  });
});

describe('integration over a tiny real run', () => {
  const ds = tictacDataset('mixed', 20, 1);
  const run = trainTrace(ds, { steps: 21, seed: 1, l1Lambda: 0.001, evalSample: 10 });
  const suite = buildProbeSuite(1, 10);

  it('computeMetrics returns finite in-range points including the final step', () => {
    const pts = computeMetrics(run, suite, 10);
    expect(pts[0].step).toBe(0);
    expect(pts[pts.length - 1].step).toBe(20);
    for (const p of pts) {
      expect(Number.isFinite(p.loss)).toBe(true);
      expect(p.agreement).toBeGreaterThanOrEqual(0);
      expect(p.agreement).toBeLessThanOrEqual(1);
      expect(p.equivariance).toBeGreaterThanOrEqual(0);
      expect(p.equivariance).toBeLessThanOrEqual(1);
      expect(p.sparsity).toBeGreaterThanOrEqual(0);
      expect(p.illegalMass).toBeGreaterThanOrEqual(0);
      expect(p.illegalMass).toBeLessThanOrEqual(1);
      expect(p.bitsVsOptimal).toBeLessThanOrEqual(10);
    }
  });

  it('sparsityReport finds the FFN and behaves at extremes', () => {
    const w = run.weights[run.weights.length - 1];
    const rep = sparsityReport(run, w, 0.01);
    expect(rep.liveUnits).toBeGreaterThan(0);
    expect(rep.liveUnits).toBeLessThanOrEqual(run.cfg.ffnHid);
    const zeros = sparsityReport(run, new Float64Array(w.length), 0.01);
    expect(zeros.frac).toBe(1);
    expect(zeros.liveUnits).toBe(0);
    expect(zeros.edges).toBe(0);
  });

  it('unitFeatureCorrelation is [ffnHid, 26] with r in [-1, 1]', () => {
    const replay = makeReplay(run);
    const corr = unitFeatureCorrelation(replay.viz, run.steps.length - 1, suite, run.cfg.ffnHid);
    expect(corr.shape).toEqual([run.cfg.ffnHid, 26]);
    for (const r of corr.data) {
      expect(r).toBeGreaterThanOrEqual(-1.000001);
      expect(r).toBeLessThanOrEqual(1.000001);
    }
  });

  it('pca2: PC1 variance ≥ PC2 variance', () => {
    const replay = makeReplay(run);
    const table = replay.tokenTable(run.steps.length - 1);
    expect(table.shape).toEqual([10, 24]);
    const pts = pca2(table);
    const varOf = (sel: (p: { x: number; y: number }) => number) => {
      const vals = pts.map(sel);
      const m = vals.reduce((a, v) => a + v, 0) / vals.length;
      return vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length;
    };
    expect(varOf((p) => p.x)).toBeGreaterThanOrEqual(varOf((p) => p.y) - 1e-9);
  });

  it('makeReplay lens returns a 3-rung ladder', () => {
    const replay = makeReplay(run);
    const report = replay.lens(5, gameIds([0, 4, 1]));
    expect(report.rungs.length).toBe(3);
    expect(report.uniformBits).toBeCloseTo(Math.log2(10), 6);
  });
});
