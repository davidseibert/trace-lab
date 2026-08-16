/**
 * The Tic·arena match machinery: seeded head-to-head games between any two
 * TicPlayers, round-robin standings, and the probe-suite report card that
 * turns "how good is this player" into the lens's usual quantities —
 * agreement with minimax, bits vs the optimal set, D₄ equivariance, illegal
 * mass, and (for LLMs) decisiveness.
 *
 * Everything async because LLM players go through the engine; toy and oracle
 * policies resolve immediately. All sampling is seeded — a (players, seed,
 * temperature) triple replays exactly.
 */

import { mulberry32, randInt, type Rng } from '../llm/rng';
import {
  analyze,
  boardFromMoves,
  isTerminal,
  legalMoves,
  transformBoard,
  transformMoves,
  TRANSFORMS,
  winner,
  type Board
} from './game';
import type { ProbePosition } from './metrics';
import { EQUIV_SUB } from './metrics';
import type { TicPlayer } from './players';

export interface MoveRecord {
  cell: number;
  /** The mover's legal-renormalized policy at this ply. */
  pi: Float64Array;
  /** Digit mass the mover put on occupied cells / '·' (before renorm). */
  illegalMass: number;
}

export interface GameRecord {
  x: string;
  o: string;
  moves: number[];
  records: MoveRecord[];
  winner: 0 | 1 | 2;
}

export interface MatchResult {
  x: string;
  o: string;
  games: GameRecord[];
  xWins: number;
  oWins: number;
  draws: number;
}

/** Legal-renormalize a player's raw 10-wide probs at a position. Zero legal
 * mass (an LLM fully off-task) falls back to uniform-over-legal with
 * illegalMass 1 — the game must go on, and the record says why. */
export function legalPolicy(probs: Float64Array, legal: number[]): { pi: Float64Array; illegalMass: number } {
  const pi = new Float64Array(9);
  let legalSum = 0;
  let total = 0;
  for (let i = 0; i < 10; i++) total += probs[i];
  for (const m of legal) {
    pi[m] = probs[m + 1];
    legalSum += pi[m];
  }
  if (legalSum > 0) {
    for (const m of legal) pi[m] /= legalSum;
    return { pi, illegalMass: Math.max(0, total - legalSum) };
  }
  for (const m of legal) pi[m] = 1 / legal.length;
  return { pi, illegalMass: 1 };
}

/** Pick from pi at a temperature: 0 = argmax (first index on exact ties),
 * else sample from pi^(1/T) renormalized — the same tempering identity the
 * distillation signal and Hopfield·heads use. */
export function pickMove(pi: Float64Array, legal: number[], rng: Rng, temperature: number): number {
  if (temperature === 0) {
    let best = legal[0];
    for (const m of legal) if (pi[m] > pi[best]) best = m;
    return best;
  }
  const w = legal.map((m) => Math.max(pi[m], 1e-12) ** (1 / temperature));
  const sum = w.reduce((a, x) => a + x, 0);
  let r = rng() * sum;
  for (let i = 0; i < legal.length; i++) {
    r -= w[i];
    if (r <= 0) return legal[i];
  }
  return legal[legal.length - 1];
}

export async function playGame(pX: TicPlayer, pO: TicPlayer, rng: Rng, temperature: number): Promise<GameRecord> {
  const moves: number[] = [];
  const records: MoveRecord[] = [];
  let b = boardFromMoves(moves);
  while (!isTerminal(b)) {
    const mover = moves.length % 2 === 0 ? pX : pO;
    const legal = legalMoves(b);
    const { pi, illegalMass } = legalPolicy(await mover.policy(moves), legal);
    const cell = pickMove(pi, legal, rng, temperature);
    records.push({ cell, pi, illegalMass });
    moves.push(cell);
    b = boardFromMoves(moves);
  }
  return { x: pX.id, o: pO.id, moves, records, winner: winner(b) };
}

export async function playMatch(
  pX: TicPlayer,
  pO: TicPlayer,
  opts: { games: number; seed: number; temperature: number; onGame?: (i: number, g: GameRecord) => void }
): Promise<MatchResult> {
  const games: GameRecord[] = [];
  let xWins = 0;
  let oWins = 0;
  let draws = 0;
  for (let i = 0; i < opts.games; i++) {
    const g = await playGame(pX, pO, mulberry32(opts.seed * 1000 + i), opts.temperature);
    games.push(g);
    if (g.winner === 1) xWins++;
    else if (g.winner === 2) oWins++;
    else draws++;
    opts.onGame?.(i, g);
  }
  return { x: pX.id, o: pO.id, games, xWins, oWins, draws };
}

/** Every ordered pair (A as X vs B as O), A ≠ B. */
export async function roundRobin(
  players: TicPlayer[],
  opts: { games: number; seed: number; temperature: number },
  onProgress?: (done: number, total: number, current: string) => void
): Promise<MatchResult[]> {
  const out: MatchResult[] = [];
  const total = players.length * (players.length - 1);
  for (const pX of players) {
    for (const pO of players) {
      if (pX.id === pO.id) continue;
      onProgress?.(out.length, total, `${pX.label} vs ${pO.label}`);
      out.push(await playMatch(pX, pO, opts));
    }
  }
  onProgress?.(total, total, 'done');
  return out;
}

// ---------------------------------------------------------------------------
// Report card — the probe suite pointed at any player
// ---------------------------------------------------------------------------

export interface ReportCard {
  agreement: number;
  bitsVsOptimal: number;
  equivariance: number;
  illegalMass: number;
  /** Mean Σ of the raw returned probs — 1 for toys/oracles; an LLM's mass on
   * ANY digit at all (the rest went to off-task tokens). */
  decisiveness: number;
  /** Fraction of forced-block positions (unique optimal move parrying an
   * opponent two-in-a-row) where the argmax finds the block — the tactical
   * column, and the one that predicts the round-robin's defense results. */
  blocks: number;
}

async function policyAtP(player: TicPlayer, p: ProbePosition) {
  const raw = await player.policy(p.moves);
  let total = 0;
  for (let i = 0; i < 10; i++) total += raw[i];
  return { ...legalPolicy(raw, p.legal), decisiveness: total };
}

/** TV distance between the pullback g⁻¹·π(g·s) and π(s) — the same
 * definition as metrics.equivError, over an async player policy. */
export async function playerEquivError(player: TicPlayer, p: ProbePosition, g: number): Promise<number> {
  const base = (await policyAtP(player, p)).pi;
  const gp: ProbePosition = {
    moves: transformMoves(g, p.moves),
    board: transformBoard(g, p.board) as Board,
    legal: transformMoves(g, p.legal),
    optimal: transformMoves(g, p.optimal)
  };
  const trans = (await policyAtP(player, gp)).pi;
  let tv = 0;
  for (const m of p.legal) tv += Math.abs(trans[TRANSFORMS[g][m]] - base[m]);
  return tv / 2;
}

export async function reportCard(
  player: TicPlayer,
  suite: ProbePosition[],
  blockSuite: ProbePosition[],
  onProgress?: (done: number, total: number) => void
): Promise<ReportCard> {
  const sub = suite.slice(0, EQUIV_SUB);
  const total = suite.length + sub.length * 7 + blockSuite.length;
  let done = 0;
  let agree = 0;
  let bits = 0;
  let illegal = 0;
  let decisive = 0;
  for (const p of suite) {
    const { pi, illegalMass, decisiveness } = await policyAtP(player, p);
    illegal += illegalMass;
    decisive += decisiveness;
    let arg = p.legal[0];
    for (const m of p.legal) if (pi[m] > pi[arg]) arg = m;
    if (p.optimal.includes(arg)) agree++;
    let optMass = 0;
    for (const m of p.optimal) optMass += pi[m];
    bits += Math.min(10, -Math.log2(Math.max(optMass, 2 ** -10)));
    onProgress?.(++done, total);
  }
  let equiv = 0;
  for (const p of sub) {
    for (let g = 1; g < 8; g++) {
      equiv += await playerEquivError(player, p, g);
      onProgress?.(++done, total);
    }
  }
  let blocked = 0;
  for (const p of blockSuite) {
    const { pi } = await policyAtP(player, p);
    let arg = p.legal[0];
    for (const m of p.legal) if (pi[m] > pi[arg]) arg = m;
    if (arg === p.optimal[0]) blocked++;
    onProgress?.(++done, total);
  }
  return {
    agreement: agree / suite.length,
    bitsVsOptimal: bits / suite.length,
    equivariance: equiv / (sub.length * 7),
    illegalMass: illegal / suite.length,
    decisiveness: decisive / suite.length,
    blocks: blockSuite.length ? blocked / blockSuite.length : 0
  };
}
