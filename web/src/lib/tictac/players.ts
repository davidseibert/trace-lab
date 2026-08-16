/**
 * The Tic·arena player abstraction: anything that maps a position to a move
 * distribution can compete — trained TicRuns, the solver (with two different
 * tie-breaking personalities), a random baseline, and real LLMs read through
 * the engine's /lens endpoint as one next-token distribution per position.
 *
 * Everything returns the 10-wide raw token-probability convention from
 * TicRun.probsAt ('·' + 9 cells). The vector need not sum to 1: an LLM player
 * returns only the mass it placed on digit tokens, so Σprobs is its
 * "decisiveness" — how much of the model's next-token belief was on-task.
 */

import { fetchLens, type TopTok } from '../logit/api';
import { randInt, type Rng } from '../llm/rng';
import { analyze, boardFromMoves, boardKey, legalMoves, toMove, type Board } from './game';
import type { Arch, Signal, TicRun } from './ticTrain';

export interface TicPlayer {
  /** 'oracle:perfect' | 'oracle:traps' | 'oracle:random' | 'toy:gpt:games' | 'llm:<model>:<mode>' */
  id: string;
  label: string;
  kind: 'toy' | 'oracle' | 'llm';
  /** 10-wide raw token probs at the position (may sum to < 1 for LLMs). */
  policy(moves: number[]): Promise<Float64Array>;
}

// ---------------------------------------------------------------------------
// Toy + oracle players
// ---------------------------------------------------------------------------

export function toyPlayer(arch: Arch, signal: Signal, run: TicRun): TicPlayer {
  const final = run.steps.length - 1;
  return {
    id: `toy:${arch}:${signal}`,
    label: `${arch}+${signal}`,
    kind: 'toy',
    policy: (moves) => Promise.resolve(run.probsAt(final, moves))
  };
}

/**
 * How many of the opponent's replies LOSE (for them) after `move` is played
 * from `b` — the tie-breaking quantity that separates minimax-equal moves.
 * (Session-verified: from the empty board, a corner leaves 7/8 losing replies,
 * the center 4/8.)
 */
export function trapCount(b: Board, move: number): number {
  const mover = toMove(b);
  const child = b.slice() as Board;
  child[move] = mover;
  let traps = 0;
  for (const r of legalMoves(child)) {
    const grand = child.slice() as Board;
    grand[r] = toMove(child);
    const v = analyze(grand).value;
    if (mover === 1 ? v === 1 : v === -1) traps++;
  }
  return traps;
}

/**
 * The solver as a player. 'uniform' spreads evenly over the minimax-optimal
 * set (the corpus generator's policy); 'traps' weights each optimal move by
 * 1 + its trap count — the same optimality, a different opponent model.
 */
export function solverPlayer(tiebreak: 'uniform' | 'traps'): TicPlayer {
  return {
    id: tiebreak === 'uniform' ? 'oracle:perfect' : 'oracle:traps',
    label: tiebreak === 'uniform' ? 'solver' : 'solver·trappy',
    kind: 'oracle',
    policy(moves) {
      const b = boardFromMoves(moves);
      const opt = analyze(b).optimal;
      const probs = new Float64Array(10);
      if (!opt.length) return Promise.resolve(probs);
      const w = opt.map((m) => (tiebreak === 'traps' ? 1 + trapCount(b, m) : 1));
      const sum = w.reduce((a, x) => a + x, 0);
      opt.forEach((m, i) => (probs[m + 1] = w[i] / sum));
      return Promise.resolve(probs);
    }
  };
}

export function randomPlayer(): TicPlayer {
  return {
    id: 'oracle:random',
    label: 'random',
    kind: 'oracle',
    policy(moves) {
      const legal = legalMoves(boardFromMoves(moves));
      const probs = new Float64Array(10);
      for (const m of legal) probs[m + 1] = 1 / legal.length;
      return Promise.resolve(probs);
    }
  };
}

// ---------------------------------------------------------------------------
// LLM players — distribution read via /lens
// ---------------------------------------------------------------------------

/** Sum digit-token variants ('3', ' 3', …) from a top-k readout into the
 * 10-wide convention. `decisiveness` = total digit mass — the rest of the
 * model's belief went to off-task tokens. */
export function parseDigitProbs(top: TopTok[]): { probs: Float64Array; decisiveness: number } {
  const probs = new Float64Array(10);
  let total = 0;
  for (const { t, p } of top) {
    const s = t.trim();
    if (s.length === 1 && s >= '0' && s <= '8') {
      probs[Number(s) + 1] += p;
      total += p;
    }
  }
  return { probs, decisiveness: total };
}

const CELL_LEGEND = '0 1 2\n3 4 5\n6 7 8';

export function renderBoardText(b: Board): string {
  const mark = (c: number) => (c === 1 ? 'X' : c === 2 ? 'O' : '.');
  return [0, 3, 6].map((r) => `${mark(b[r])} ${mark(b[r + 1])} ${mark(b[r + 2])}`).join('\n');
}

/** The prompt whose NEXT token is the move digit. Ends without trailing
 * space, so tokenizers that fuse the space into the digit (' 8') still land
 * on a single digit token — parseDigitProbs trims either way. */
export function movePrompt(moves: number[], chat: boolean): string {
  const b = boardFromMoves(moves);
  const mover = toMove(b) === 1 ? 'X' : 'O';
  const base =
    `Tic-tac-toe. Cells are numbered 0-8, left to right, top to bottom:\n${CELL_LEGEND}\n` +
    `Current board (X, O, . = empty):\n${renderBoardText(b)}\n` +
    `It is ${mover}'s turn.`;
  return chat
    ? `${base}\nReply with only the digit of the best cell for ${mover} to play.`
    : `${base} The single best cell number for ${mover} to play is`;
}

/**
 * Chat templating is available, but measured decisiveness says raw completion
 * is the reliable default: Qwen3-0.6B chat answers "The…" (digit mass ~0.1%),
 * while raw with the right ending hits 94–100% on Qwen3/Gemma.
 */
export function defaultLlmMode(_model: string): 'chat' | 'raw' {
  return 'raw';
}

export function llmPlayer(model: string, mode: 'chat' | 'raw'): TicPlayer {
  // Positions repeat constantly across games and the transformed probe suite;
  // one engine forward per UNIQUE position is the whole cost model.
  const memo = new Map<number, Promise<Float64Array>>();
  // Tokenizer-dependent prompt ending, self-calibrated on the first call:
  // GPT-2-style BPEs fuse " 4" into one token (want NO trailing space), while
  // Qwen/Gemma emit the space separately (want the trailing space). Measured:
  // Qwen3 raw goes 0% → 94% decisive with the space; gpt2 the reverse.
  let ending: '' | ' ' | null = null;

  const query = async (moves: number[], end: '' | ' ') => {
    const r = await fetchLens({
      model,
      prompt: movePrompt(moves, mode === 'chat') + (mode === 'raw' ? end : ''),
      top_k: 20,
      jlens: false,
      rollout: 0,
      chat: mode === 'chat',
      thinking: false
    });
    const finalLayer = r.grid[r.grid.length - 1];
    return parseDigitProbs(finalLayer[finalLayer.length - 1]);
  };

  return {
    id: `llm:${model}:${mode}`,
    label: `${model.split('/').pop()} (${mode})`,
    kind: 'llm',
    policy(moves) {
      const key = boardKey(boardFromMoves(moves));
      const hit = memo.get(key);
      if (hit) return hit;
      const p = (async () => {
        if (ending === null && mode === 'raw') {
          const bare = await query(moves, '');
          if (bare.decisiveness >= 0.05) {
            ending = '';
            return bare.probs;
          }
          const spaced = await query(moves, ' ');
          ending = spaced.decisiveness > bare.decisiveness ? ' ' : '';
          return (ending === ' ' ? spaced : bare).probs;
        }
        return (await query(moves, ending ?? '')).probs;
      })();
      memo.set(key, p);
      p.catch(() => memo.delete(key)); // don't cache failures
      return p;
    }
  };
}
