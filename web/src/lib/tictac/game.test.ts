import { describe, expect, it } from 'bun:test';
import { mulberry32, randInt } from '../llm/rng';
import {
  allStates,
  analyze,
  boardFromMoves,
  boardKey,
  emptyBoard,
  generateCorpus,
  generateGame,
  INVERSE,
  isTerminal,
  legalMoves,
  LINES,
  orbitCount,
  toMove,
  transformBoard,
  transformMoves,
  TRANSFORMS,
  winner,
  type Board
} from './game';

describe('D₄ group', () => {
  it('has 8 distinct bijective transforms', () => {
    const keys = new Set(TRANSFORMS.map((t) => t.join(',')));
    expect(keys.size).toBe(8);
    for (const t of TRANSFORMS) expect([...t].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('is closed under composition', () => {
    const table = new Set(TRANSFORMS.map((t) => t.join(',')));
    for (const a of TRANSFORMS)
      for (const b of TRANSFORMS) {
        const comp = a.map((_, i) => b[a[i]]);
        expect(table.has(comp.join(','))).toBe(true);
      }
  });

  it('INVERSE is correct and r90⁴ = e', () => {
    for (let g = 0; g < 8; g++) {
      const inv = INVERSE[g];
      for (let i = 0; i < 9; i++) expect(TRANSFORMS[inv][TRANSFORMS[g][i]]).toBe(i);
    }
    let p = [...TRANSFORMS[0]];
    for (let k = 0; k < 4; k++) p = p.map((_, i) => TRANSFORMS[1][p[i]]);
    expect(p).toEqual([...TRANSFORMS[0]]);
  });

  it('maps win lines onto win lines', () => {
    const lineSet = new Set(LINES.map((l) => [...l].sort((a, b) => a - b).join(',')));
    for (const t of TRANSFORMS)
      for (const l of LINES) {
        const img = l.map((c) => t[c]).sort((a, b) => a - b).join(',');
        expect(lineSet.has(img)).toBe(true);
      }
  });
});

describe('state space', () => {
  it('has exactly 5,478 legal positions', () => {
    expect(allStates().length).toBe(5478);
  });

  it('has exactly 765 orbits under D₄', () => {
    expect(orbitCount()).toBe(765);
  });
});

describe('solver', () => {
  it('empty board: draw value, all 9 openings optimal, every outcome achievable', () => {
    const a = analyze(emptyBoard());
    expect(a.value).toBe(0);
    expect([...a.optimal].sort((x, y) => x - y)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(a.achievable).toBe(7);
  });

  it('forced block: after X0 O4 X1, O must play 2', () => {
    expect(analyze(boardFromMoves([0, 4, 1])).optimal).toEqual([2]);
  });

  it('opposite-corner trap: after X0 O4 X8, O must avoid corners', () => {
    const opt = analyze(boardFromMoves([0, 4, 8])).optimal;
    expect(opt.length).toBeGreaterThan(0);
    for (const m of opt) expect([1, 3, 5, 7]).toContain(m);
  });

  it('is D₄-equivariant: optimal(g·s) = g·optimal(s), value invariant', () => {
    const rng = mulberry32(99);
    for (let trial = 0; trial < 50; trial++) {
      // random non-terminal position from a random-play prefix
      const game = generateGame(mulberry32(trial + 1), 'random');
      const cut = randInt(rng, game.length);
      const b = boardFromMoves(game.slice(0, cut));
      if (isTerminal(b)) continue;
      const a = analyze(b);
      for (let g = 0; g < 8; g++) {
        const ag = analyze(transformBoard(g, b) as Board);
        expect(ag.value).toBe(a.value);
        expect(ag.achievable).toBe(a.achievable);
        expect([...ag.optimal].sort((x, y) => x - y)).toEqual(
          transformMoves(g, a.optimal).sort((x, y) => x - y)
        );
      }
    }
  });

  it('perfect play always draws', () => {
    for (let s = 1; s <= 50; s++) {
      const game = generateGame(mulberry32(s), 'optimal');
      const b = boardFromMoves(game);
      expect(game.length).toBe(9);
      expect(winner(b)).toBe(0);
    }
  });

  it('achievable outcomes shrink monotonically along any game', () => {
    for (let s = 1; s <= 20; s++) {
      for (const policy of ['optimal', 'random'] as const) {
        const game = generateGame(mulberry32(s * 7 + (policy === 'random' ? 1 : 0)), policy);
        let prev = analyze(emptyBoard()).achievable;
        for (let k = 1; k <= game.length; k++) {
          const cur = analyze(boardFromMoves(game.slice(0, k))).achievable;
          expect(cur & ~prev).toBe(0); // no outcome comes back
          prev = cur;
        }
      }
    }
  });
});

describe('board mechanics + corpus', () => {
  it('boardFromMoves validates', () => {
    expect(() => boardFromMoves([0, 0])).toThrow();
    expect(() => boardFromMoves([9])).toThrow();
    expect(() => boardFromMoves([0, 3, 1, 4, 2, 5])).toThrow(); // X wins at move 5; O's 6th is illegal
  });

  it('toMove alternates and legalMoves excludes finished games', () => {
    expect(toMove(emptyBoard())).toBe(1);
    expect(toMove(boardFromMoves([4]))).toBe(2);
    const won = boardFromMoves([0, 3, 1, 4, 2]); // X wins top row
    expect(winner(won)).toBe(1);
    expect(legalMoves(won)).toEqual([]);
  });

  it('boardKey is injective on a sample and transformBoard round-trips', () => {
    const b = boardFromMoves([4, 0, 8, 2]);
    for (let g = 0; g < 8; g++) {
      const there = transformBoard(g, b);
      const back = transformBoard(INVERSE[g], there);
      expect(boardKey(back)).toBe(boardKey(b));
    }
  });

  it('corpus is deterministic, legal, and token-shaped', () => {
    const a = generateCorpus('mixed', 30, 5);
    const b = generateCorpus('mixed', 30, 5);
    expect(a).toEqual(b);
    for (const seq of a) {
      expect(seq[0]).toBe('·');
      expect(seq.length).toBeLessThanOrEqual(10);
      boardFromMoves(seq.slice(1).map(Number)); // throws if illegal
    }
    expect(generateCorpus('mixed', 30, 6)).not.toEqual(a);
  });
});
