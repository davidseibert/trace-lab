import { describe, expect, it } from 'bun:test';
import { trainTrace } from '../llm/trainTrace';
import { tictacDataset } from './dataset';
import { ticTrain, gameIds, type Arch, type Signal, type TicRun } from './ticTrain';
import { analyze, emptyBoard } from './game';

const SMALL = { kind: 'mixed' as const, nGames: 20, steps: 21, seed: 1, l1Lambda: 0.001 };

describe('gpt+games parity with trainTrace', () => {
  it('reproduces the trainTrace loss curve and weights bit-for-bit', () => {
    const legacy = trainTrace(tictacDataset('mixed', 20, 1), { steps: 21, seed: 1, l1Lambda: 0.001, evalSample: 50 });
    const run = ticTrain({ ...SMALL, arch: 'gpt', signal: 'games' });
    expect(run.steps.map((s) => s.loss)).toEqual(legacy.steps.map((s) => s.loss));
    expect([...run.weights[20]]).toEqual([...legacy.weights[20]]);
  });
});

describe('the grid', () => {
  const teacher = ticTrain({ ...SMALL, arch: 'gpt', signal: 'games' });
  const cells: [Arch, Signal][] = [
    ['gpt', 'solver'],
    ['encoder', 'games'],
    ['encoder', 'solver'],
    ['encoder', 'distill'],
    ['mlp', 'games'],
    ['mlp', 'solver'],
    ['mlp', 'distill']
  ];

  for (const [arch, signal] of cells) {
    it(`${arch}+${signal} trains with finite falling loss and a sane adapter`, () => {
      // 101 steps: enough for the loss to fall reliably under batch-1 noise.
      const run = ticTrain({ ...SMALL, steps: 101, arch, signal, temperature: 2, teacher });
      expect(run.steps.length).toBe(101);
      for (const s of run.steps) expect(Number.isFinite(s.loss)).toBe(true);
      expect(run.steps[100].loss).toBeLessThan(run.steps[0].loss);

      const probs = run.probsAt(20, [0, 4]);
      expect(probs.length).toBe(10);
      let sum = 0;
      for (const p of probs) sum += p;
      if (arch === 'gpt') expect(sum).toBeCloseTo(1, 9);
      else {
        expect(probs[0]).toBe(0); // board archs cannot emit '·'
        expect(sum).toBeCloseTo(1, 9);
      }

      const units = run.unitsAt(20, [0, 4]);
      expect(units.length).toBe(run.meta.nUnits);

      const sp = run.sparsity(20, 0.01);
      expect(sp.liveUnits).toBeGreaterThan(0);
      expect(sp.liveUnits).toBeLessThanOrEqual(run.meta.nUnits);

      const table = run.tokenTable(20);
      expect(table.shape[0]).toBe(arch === 'gpt' ? 10 : 9);

      expect(run.lensAt(20, [0, 4]) === null).toBe(arch !== 'gpt');
    });
  }

  it('distill requires a teacher and rejects a gpt student', () => {
    expect(() => ticTrain({ ...SMALL, arch: 'mlp', signal: 'distill' })).toThrow();
    expect(() => ticTrain({ ...SMALL, arch: 'gpt', signal: 'distill', teacher })).toThrow();
  });

  it('determinism: same options ⇒ identical runs', () => {
    const a = ticTrain({ ...SMALL, arch: 'encoder', signal: 'solver' });
    const b = ticTrain({ ...SMALL, arch: 'encoder', signal: 'solver' });
    expect([...a.weights[20]]).toEqual([...b.weights[20]]);
  });
});

describe('solver-soft signal', () => {
  it('produces a flatter opening than sampled games (the wobble cure)', () => {
    // The comparative claim is the real one: same arch/steps/seed, the soft
    // solver target (exactly uniform at ply 0) yields a lower-variance opening
    // distribution than the sampled-games estimator of the same objective.
    expect(analyze(emptyBoard()).optimal.length).toBe(9);
    const opts = { kind: 'optimal' as const, nGames: 50, steps: 301, seed: 1, l1Lambda: 0, arch: 'mlp' as const };
    const std = (run: TicRun) => {
      const p = [...run.probsAt(300, []).slice(1)];
      const m = p.reduce((a, v) => a + v, 0) / 9;
      return Math.sqrt(p.reduce((a, v) => a + (v - m) ** 2, 0) / 9);
    };
    const soft = std(ticTrain({ ...opts, signal: 'solver' }));
    const hard = std(ticTrain({ ...opts, signal: 'games' }));
    expect(soft).toBeLessThan(hard);
    expect(soft).toBeLessThan(0.06); // uniform is 0.111 per cell; stay near it
  });
});
