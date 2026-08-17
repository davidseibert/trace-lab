/**
 * Freeze the arena's gpt+solver toy into a reward-model lookup table — the
 * teacher for the mapped-Goodhart RLVR arm (INSIGHTS §5.9).
 *
 * Trains the exact cell the Tic·arena trains (arch gpt, signal solver,
 * kind optimal, 400 games, 400 steps, seed 1, λ 0.001 — ArenaApp.svelte's
 * TOY_* constants), then walks every reachable position depth-first (moves in
 * ascending cell order, first visit wins) and records the toy's raw
 * next-token mass on the nine cells at that position's canonical move
 * sequence. The toy is sequence-dependent; the table freezes one sequence per
 * board, and from then on the TABLE is the reward model — its error map
 * against the solver is exact by construction, which is the whole point:
 * Goodhart with the cliff mapped before training starts.
 *
 *   cd web && bun run scripts/export-teacher.ts
 *
 * Output: engine/data/local/tic-teacher-gpt-solver.json
 *   { meta: {...}, policies: { "<boardKey>": [9 × cell mass] } }
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { ticTrain } from '../src/lib/tictac/ticTrain';
import { boardFromMoves, boardKey, isTerminal, legalMoves, toMove, type Board } from '../src/lib/tictac/game';

const OPTS = { arch: 'gpt', signal: 'solver', kind: 'optimal', nGames: 400, steps: 400, seed: 1, l1Lambda: 0.001 } as const;

console.error(`training ${OPTS.arch}+${OPTS.signal}…`);
const run = ticTrain({ ...OPTS });
const final = run.steps.length - 1;

const policies: Record<string, number[]> = {};
let visited = 0;

function walk(moves: number[]): void {
  const b = boardFromMoves(moves);
  const key = String(boardKey(b));
  if (key in policies) return;
  if (isTerminal(b)) return;
  const probs = run.probsAt(final, moves); // 10-wide, '·' + 9 cells
  policies[key] = [...probs.slice(1)].map((p) => Number(p.toFixed(6)));
  visited++;
  for (const m of legalMoves(b)) walk([...moves, m]);
}
walk([]);

const out = join(import.meta.dir, '../../engine/data/local/tic-teacher-gpt-solver.json');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ meta: { ...OPTS, params: run.meta.paramCount, positions: visited }, policies }));
console.error(`wrote ${visited} positions (${run.meta.paramCount} params) → ${out}`);
