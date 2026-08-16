/**
 * Tic-tac-toe, solved — the ground truth every Tic·tac panel is checked
 * against. Pure math, no Svelte imports.
 *
 * Three exact structures live here, and the lens leans on all of them:
 *  - the full minimax solution (5,478 legal positions, memoized), so "the
 *    optimal move" is a computed fact, never a heuristic;
 *  - the board's symmetry group D₄ (8 transforms → 765 orbits), so
 *    equivariance of a learned policy is measurable against a known action;
 *  - the outcome poset: position → set of still-achievable outcomes, which
 *    only ever shrinks as the game descends its DAG.
 *
 * Board cells are indexed 0..8 row-major:  0 1 2 / 3 4 5 / 6 7 8.  X moves
 * first. Everything random funnels through a caller-supplied seeded Rng.
 */

import { mulberry32, randInt, type Rng } from '../llm/rng';

/** 0 = empty, 1 = X, 2 = O. */
export type Cell = 0 | 1 | 2;
export type Board = Cell[];

/** Outcome bitmask: X-win | draw | O-win. */
export const X_WIN = 1;
export const DRAW = 2;
export const O_WIN = 4;

/** The 8 win lines: 3 rows, 3 columns, 2 diagonals. */
export const LINES: readonly number[][] = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

/**
 * The dihedral group D₄ acting on cells: TRANSFORMS[g][i] = image of cell i.
 * A move m maps to TRANSFORMS[g][m]; a board maps by B'[T[g][i]] = B[i].
 * Order: identity, three rotations (90° CW steps), then the four mirrors.
 * The unit tests re-derive closure/inverses, so a transcription slip here
 * fails loudly rather than silently skewing every equivariance number.
 */
export const TRANSFORMS: readonly number[][] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8], // e
  [2, 5, 8, 1, 4, 7, 0, 3, 6], // r90 (CW)
  [8, 7, 6, 5, 4, 3, 2, 1, 0], // r180
  [6, 3, 0, 7, 4, 1, 8, 5, 2], // r270
  [2, 1, 0, 5, 4, 3, 8, 7, 6], // flipH (mirror across the vertical axis)
  [6, 7, 8, 3, 4, 5, 0, 1, 2], // flipV (mirror across the horizontal axis)
  [0, 3, 6, 1, 4, 7, 2, 5, 8], // diag (transpose along 0-4-8)
  [8, 5, 2, 7, 4, 1, 6, 3, 0]  // anti (transpose along 2-4-6)
];

/** INVERSE[g] = index of g⁻¹ (rotations pair up; every mirror is its own). */
export const INVERSE: readonly number[] = [0, 3, 2, 1, 4, 5, 6, 7];

export function transformBoard(g: number, b: Board): Board {
  const t = TRANSFORMS[g];
  const out: Board = new Array(9).fill(0) as Board;
  for (let i = 0; i < 9; i++) out[t[i]] = b[i];
  return out;
}

export function transformMoves(g: number, moves: number[]): number[] {
  const t = TRANSFORMS[g];
  return moves.map((m) => t[m]);
}

/** Base-3 integer encoding, 0..3⁹−1 — the memo/orbit key. */
export function boardKey(b: Board): number {
  let k = 0;
  for (let i = 8; i >= 0; i--) k = k * 3 + b[i];
  return k;
}

/** Canonical representative: the minimal boardKey over the orbit, with the
 * witnessing transform. */
export function canonical(b: Board): { key: number; g: number } {
  let best = Infinity;
  let bestG = 0;
  for (let g = 0; g < 8; g++) {
    const k = boardKey(transformBoard(g, b));
    if (k < best) {
      best = k;
      bestG = g;
    }
  }
  return { key: best, g: bestG };
}

export function emptyBoard(): Board {
  return new Array(9).fill(0) as Board;
}

/** Whose turn: X when mark counts are equal (X always moves first). */
export function toMove(b: Board): 1 | 2 {
  let x = 0;
  let o = 0;
  for (const c of b) {
    if (c === 1) x++;
    else if (c === 2) o++;
  }
  return x === o ? 1 : 2;
}

export function winner(b: Board): 0 | 1 | 2 {
  for (const [a, m, z] of LINES) {
    if (b[a] !== 0 && b[a] === b[m] && b[a] === b[z]) return b[a] as 1 | 2;
  }
  return 0;
}

export function isTerminal(b: Board): boolean {
  return winner(b) !== 0 || b.every((c) => c !== 0);
}

export function legalMoves(b: Board): number[] {
  if (winner(b) !== 0) return [];
  const out: number[] = [];
  for (let i = 0; i < 9; i++) if (b[i] === 0) out.push(i);
  return out;
}

/** Replay a move list from the empty board; throws on any illegal move
 * (occupied cell, out of range, or a move after the game ended). */
export function boardFromMoves(moves: number[]): Board {
  const b = emptyBoard();
  for (const m of moves) {
    if (!Number.isInteger(m) || m < 0 || m > 8) throw new Error(`bad move ${m}`);
    if (b[m] !== 0) throw new Error(`cell ${m} already taken`);
    if (winner(b) !== 0) throw new Error(`move ${m} after game over`);
    b[m] = toMove(b);
  }
  return b;
}

export interface Analysis {
  /** Minimax value from X's perspective: 1 X wins, 0 draw, −1 O wins. */
  value: -1 | 0 | 1;
  /** Every move for the player to move that achieves `value` (empty at a
   * terminal position). */
  optimal: number[];
  /** Bitmask of outcomes reachable under ANY legal continuation — monotone
   * shrinking along the game DAG by construction (OR over children). */
  achievable: number;
}

// Memoized over RAW boardKey, not the canonical orbit key — 5,478 entries is
// nothing, and keeping the solver blind to the symmetry code makes the
// "optimal(g·s) = g·optimal(s)" test a genuine cross-check, not a tautology.
const MEMO = new Map<number, Analysis>();

export function analyze(b: Board): Analysis {
  const key = boardKey(b);
  const hit = MEMO.get(key);
  if (hit) return hit;

  let result: Analysis;
  const w = winner(b);
  if (w !== 0) {
    result = { value: w === 1 ? 1 : -1, optimal: [], achievable: w === 1 ? X_WIN : O_WIN };
  } else if (b.every((c) => c !== 0)) {
    result = { value: 0, optimal: [], achievable: DRAW };
  } else {
    const mover = toMove(b);
    let best: -1 | 0 | 1 = mover === 1 ? -1 : 1;
    let achievable = 0;
    const values: (-1 | 0 | 1)[] = [];
    const moves = legalMoves(b);
    for (const m of moves) {
      const child = b.slice() as Board;
      child[m] = mover;
      const a = analyze(child);
      values.push(a.value);
      achievable |= a.achievable;
      if (mover === 1 ? a.value > best : a.value < best) best = a.value;
    }
    result = {
      value: best,
      optimal: moves.filter((_, i) => values[i] === best),
      achievable
    };
  }
  MEMO.set(key, result);
  return result;
}

/** Every legal reachable position's boardKey (terminal included) — DFS from
 * the empty board, not expanding past terminals. Exactly 5,478. */
export function allStates(): number[] {
  const seen = new Set<number>();
  const walk = (b: Board) => {
    const k = boardKey(b);
    if (seen.has(k)) return;
    seen.add(k);
    if (isTerminal(b)) return;
    const mover = toMove(b);
    for (const m of legalMoves(b)) {
      const child = b.slice() as Board;
      child[m] = mover;
      walk(child);
    }
  };
  walk(emptyBoard());
  return [...seen];
}

/** Number of D₄ orbits over all legal positions — exactly 765. */
export function orbitCount(): number {
  const orbits = new Set<number>();
  for (const k of allStates()) {
    // decode base-3 key back to a board
    const b = emptyBoard();
    let v = k;
    for (let i = 0; i < 9; i++) {
      b[i] = (v % 3) as Cell;
      v = Math.floor(v / 3);
    }
    orbits.add(canonical(b).key);
  }
  return orbits.size;
}

export type CorpusKind = 'optimal' | 'random' | 'mixed';

/** One full game to a terminal position. `optimal` samples uniformly among
 * the minimax-optimal moves for BOTH sides (note: at the empty board all
 * nine openings are optimal — perfect play is a draw from everywhere). */
export function generateGame(rng: Rng, policy: 'optimal' | 'random'): number[] {
  const b = emptyBoard();
  const moves: number[] = [];
  while (!isTerminal(b)) {
    const pool = policy === 'optimal' ? analyze(b).optimal : legalMoves(b);
    const m = pool[randInt(rng, pool.length)];
    b[m] = toMove(b);
    moves.push(m);
  }
  return moves;
}

/** Seeded corpus of token sequences ['·','4','0',…] for the Dataset. */
export function generateCorpus(kind: CorpusKind, nGames: number, seed: number): string[][] {
  const rng = mulberry32(seed);
  const out: string[][] = [];
  for (let i = 0; i < nGames; i++) {
    const policy = kind === 'mixed' ? (rng() < 0.5 ? 'optimal' : 'random') : kind;
    out.push(['·', ...generateGame(rng, policy).map(String)]);
  }
  return out;
}
