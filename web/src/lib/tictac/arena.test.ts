import { describe, expect, it } from 'bun:test';
import { mulberry32 } from '../llm/rng';
import { emptyBoard, boardFromMoves } from './game';
import { buildProbeSuite } from './metrics';
import { ticTrain } from './ticTrain';
import {
  defaultLlmMode,
  movePrompt,
  parseDigitProbs,
  randomPlayer,
  renderBoardText,
  solverPlayer,
  toyPlayer,
  trapCount
} from './players';
import { playGame, playMatch, reportCard, roundRobin } from './arena';

const perfect = solverPlayer('uniform');
const trappy = solverPlayer('traps');
const rand = randomPlayer();

describe('oracle players', () => {
  it('trap counts match the session-verified numbers', () => {
    expect(trapCount(emptyBoard(), 0)).toBe(7); // corner
    expect(trapCount(emptyBoard(), 4)).toBe(4); // center
    expect(trapCount(emptyBoard(), 1)).toBe(4); // edge
  });

  it('solver vs solver always draws', async () => {
    for (let s = 1; s <= 5; s++) {
      const g = await playGame(perfect, perfect, mulberry32(s), 1);
      expect(g.winner).toBe(0);
      const gt = await playGame(trappy, trappy, mulberry32(s), 1);
      expect(gt.winner).toBe(0);
    }
  });

  it('solver never loses to random, either color', async () => {
    const asX = await playMatch(perfect, rand, { games: 20, seed: 1, temperature: 1 });
    expect(asX.oWins).toBe(0);
    const asO = await playMatch(rand, perfect, { games: 20, seed: 2, temperature: 1 });
    expect(asO.xWins).toBe(0);
  });

  it('trappy converts at least as many wins from random as uniform does', async () => {
    const t = await playMatch(trappy, rand, { games: 100, seed: 7, temperature: 1 });
    const u = await playMatch(perfect, rand, { games: 100, seed: 7, temperature: 1 });
    expect(t.xWins).toBeGreaterThanOrEqual(u.xWins);
  });
});

describe('match mechanics', () => {
  it('temperature 0 is deterministic; temperature 1 varies with seed', async () => {
    const a = await playGame(trappy, rand, mulberry32(3), 0);
    const b = await playGame(trappy, rand, mulberry32(3), 0);
    expect(a.moves).toEqual(b.moves);
    const c = await playGame(rand, rand, mulberry32(4), 1);
    const d = await playGame(rand, rand, mulberry32(5), 1);
    expect(c.moves.join('')).not.toBe(d.moves.join(''));
  });

  it('aggregation sums to games and records are per-ply', async () => {
    const m = await playMatch(rand, rand, { games: 12, seed: 9, temperature: 1 });
    expect(m.xWins + m.oWins + m.draws).toBe(12);
    for (const g of m.games) {
      expect(g.records.length).toBe(g.moves.length);
      boardFromMoves(g.moves); // legal throughout or throws
    }
  });

  it('roundRobin covers all ordered pairs', async () => {
    const results = await roundRobin([perfect, trappy, rand], { games: 2, seed: 1, temperature: 1 });
    expect(results.length).toBe(6);
    const pairs = new Set(results.map((r) => `${r.x}|${r.o}`));
    expect(pairs.size).toBe(6);
  });
});

describe('LLM plumbing (pure parts)', () => {
  it('parseDigitProbs sums variants, ignores junk, reports decisiveness', () => {
    const { probs, decisiveness } = parseDigitProbs([
      { t: '3', p: 0.4 },
      { t: ' 3', p: 0.1 },
      { t: '8', p: 0.2 },
      { t: 'the', p: 0.2 },
      { t: '⏎', p: 0.05 },
      { t: '9', p: 0.05 } // not a cell
    ]);
    expect(probs[4]).toBeCloseTo(0.5, 9); // cell 3
    expect(probs[9]).toBeCloseTo(0.2, 9); // cell 8
    expect(decisiveness).toBeCloseTo(0.7, 9);
  });

  it('movePrompt renders the board and ends ready for a digit', () => {
    const p = movePrompt([0, 4], false);
    expect(p).toContain('X . .');
    expect(p).toContain('. O .');
    expect(p).toContain("It is X's turn.");
    expect(p.endsWith('is')).toBe(true);
    expect(renderBoardText(boardFromMoves([0, 4]))).toBe('X . .\n. O .\n. . .');
  });

  it('defaultLlmMode is raw (measured: chat answers "The…" on small instruct models)', () => {
    expect(defaultLlmMode('Qwen/Qwen3-0.6B')).toBe('raw');
    expect(defaultLlmMode('gpt2')).toBe('raw');
  });
});

describe('report card', () => {
  it('solver: perfect agreement, zero equivariance error, zero illegal', async () => {
    const card = await reportCard(perfect, buildProbeSuite(3, 16));
    expect(card.agreement).toBe(1);
    expect(card.equivariance).toBeLessThan(1e-9);
    expect(card.illegalMass).toBeLessThan(1e-9);
    expect(card.decisiveness).toBeCloseTo(1, 9);
  });

  it('toy player wraps a run and plays legal full games', async () => {
    const run = ticTrain({ arch: 'mlp', signal: 'games', kind: 'mixed', nGames: 20, steps: 21, seed: 1, l1Lambda: 0 });
    const toy = toyPlayer('mlp', 'games', run);
    const g = await playGame(toy, perfect, mulberry32(1), 0);
    expect([0, 1, 2]).toContain(g.winner);
    expect(g.winner).not.toBe(1); // nothing beats the solver
    const card = await reportCard(toy, buildProbeSuite(3, 12));
    expect(card.agreement).toBeGreaterThanOrEqual(0);
    expect(card.decisiveness).toBeCloseTo(1, 6);
  });
});
