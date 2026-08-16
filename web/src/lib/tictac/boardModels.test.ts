import { describe, expect, it } from 'bun:test';
import { mulberry32 } from '../llm/rng';
import { backward, crossEntropyLoss } from '../llm/tensor';
import { Adam } from '../llm/model';
import { makeBoardModel, paramCount, ENC_HEADS } from './boardModels';
import { boardFromMoves } from './game';

const CELLS = boardFromMoves([0, 4, 1]); // X0 O4 X1 — the forced-block position

describe('board models', () => {
  it('shapes and viz are correct for both archs', () => {
    for (const arch of ['encoder', 'mlp'] as const) {
      const m = makeBoardModel(arch, mulberry32(1));
      const logits = m.forward(CELLS);
      expect(logits.shape).toEqual([1, 9]);
      const viz = m.viz!;
      expect(viz.logits.length).toBe(9);
      let s = 0;
      for (const p of viz.probs) s += p;
      expect(s).toBeCloseTo(1, 9);
      if (arch === 'encoder') {
        expect(viz.attn!.shape).toEqual([ENC_HEADS, 9, 9]);
        expect(viz.cellEmb!.shape).toEqual([9, 24]);
        expect(viz.hidden.shape).toEqual([9, 48]);
        // Attention rows are distributions over ALL 9 cells — bidirectional.
        for (let r = 0; r < ENC_HEADS * 9; r++) {
          let rs = 0;
          for (let c = 0; c < 9; c++) rs += viz.attn!.data[r * 9 + c];
          expect(rs).toBeCloseTo(1, 9);
        }
      } else {
        expect(viz.attn).toBeNull();
        expect(viz.hidden.shape).toEqual([1, 150]);
      }
    }
  });

  it('is parameter-matched to the GPT arm (~5.5k ± 15%)', () => {
    for (const arch of ['encoder', 'mlp'] as const) {
      const n = paramCount(makeBoardModel(arch, mulberry32(1)));
      expect(n).toBeGreaterThan(4600);
      expect(n).toBeLessThan(6400);
    }
  });

  it('is deterministic given the seed', () => {
    for (const arch of ['encoder', 'mlp'] as const) {
      const a = makeBoardModel(arch, mulberry32(7));
      const b = makeBoardModel(arch, mulberry32(7));
      a.forward(CELLS);
      b.forward(CELLS);
      expect([...a.viz!.logits]).toEqual([...b.viz!.logits]);
    }
  });

  it('trains: a few Adam steps reduce loss on one example', () => {
    for (const arch of ['encoder', 'mlp'] as const) {
      const m = makeBoardModel(arch, mulberry32(3));
      const opt = new Adam(m.params(), 0.01);
      const lossAt = () => crossEntropyLoss(m.forward(CELLS), [2]).data[0];
      const before = lossAt();
      for (let i = 0; i < 20; i++) {
        opt.zeroGrad();
        backward(crossEntropyLoss(m.forward(CELLS), [2]));
        opt.step();
      }
      expect(lossAt()).toBeLessThan(before);
    }
  });
});
