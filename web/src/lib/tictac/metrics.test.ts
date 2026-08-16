import { describe, expect, it } from 'bun:test';
import { ticTrain, gameIds, type TicRun } from './ticTrain';
import { analyze, boardFromMoves, INVERSE, isTerminal, TRANSFORMS } from './game';
import {
  buildProbeSuite,
  computeMetrics,
  equivError,
  groundTruthFeatures,
  klBits,
  pca2,
  policyAt,
  unitFeatureCorrelation,
  type ProbePosition
} from './metrics';

/** A minimal TicRun whose policy is the solver's uniform-over-optimal — a
 * perfectly D₄-equivariant reference model (the solver is provably
 * equivariant; game.test.ts pins that). */
function solverRun(): TicRun {
  const probsAt = (_step: number, moves: number[]): Float64Array => {
    const b = boardFromMoves(moves);
    const a = analyze(b);
    const probs = new Float64Array(10);
    for (const m of a.optimal) probs[m + 1] = 1 / a.optimal.length;
    return probs;
  };
  return {
    arch: 'mlp',
    signal: 'solver',
    steps: [{ index: 0, loss: 0 }],
    weights: [new Float64Array(1)],
    cfg: null,
    meta: { nHeads: 0, nUnits: 1, paramCount: 1 },
    probsAt,
    unitsAt: () => new Float64Array(1),
    vizAt: () => {
      throw new Error('not needed');
    },
    lensAt: () => null,
    tokenTable: () => ({ data: new Float64Array(9), shape: [9, 1] }),
    sparsity: () => ({ frac: 0, liveUnits: 1, edges: 2 })
  };
}

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

describe('equivariance metric (through the TicRun interface)', () => {
  const run = solverRun();
  const suite = buildProbeSuite(2, 8);

  it('scores ≈ 0 for the solver policy', () => {
    for (const p of suite) for (let g = 1; g < 8; g++) expect(equivError(run, 0, p, g)).toBeLessThan(1e-9);
  });

  it('pullback indexing round-trips under INVERSE', () => {
    for (let g = 0; g < 8; g++)
      for (let m = 0; m < 9; m++) expect(TRANSFORMS[INVERSE[g]][TRANSFORMS[g][m]]).toBe(m);
  });

  it('policyAt renormalizes over legal moves and klBits(self, self) = 0', () => {
    const p = suite[0];
    const { pi } = policyAt(run, 0, p);
    let sum = 0;
    for (const m of p.legal) sum += pi[m];
    expect(sum).toBeCloseTo(1, 9);
    expect(klBits(run, run, p)).toBeCloseTo(0, 9);
  });
});

describe('integration over a tiny real run', () => {
  const run = ticTrain({ arch: 'gpt', signal: 'games', kind: 'mixed', nGames: 20, steps: 21, seed: 1, l1Lambda: 0.001 });
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

  it('unitFeatureCorrelation is [nUnits, 26] with r in [-1, 1]', () => {
    const corr = unitFeatureCorrelation(run, run.steps.length - 1, suite);
    expect(corr.shape).toEqual([run.meta.nUnits, 26]);
    for (const r of corr.data) {
      expect(r).toBeGreaterThanOrEqual(-1.000001);
      expect(r).toBeLessThanOrEqual(1.000001);
    }
  });

  it('pca2: PC1 variance ≥ PC2 variance on the token table', () => {
    const table = run.tokenTable(run.steps.length - 1);
    expect(table.shape).toEqual([10, 24]);
    const pts = pca2(table);
    const varOf = (sel: (p: { x: number; y: number }) => number) => {
      const vals = pts.map(sel);
      const m = vals.reduce((a, v) => a + v, 0) / vals.length;
      return vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length;
    };
    expect(varOf((p) => p.x)).toBeGreaterThanOrEqual(varOf((p) => p.y) - 1e-9);
  });

  it('lensAt returns a 3-rung ladder for the gpt arm', () => {
    const report = run.lensAt(5, [0, 4, 1])!;
    expect(report.rungs.length).toBe(3);
    expect(report.uniformBits).toBeCloseTo(Math.log2(10), 6);
    expect(gameIds([0, 4, 1])).toEqual([0, 1, 5, 2]);
  });
});
