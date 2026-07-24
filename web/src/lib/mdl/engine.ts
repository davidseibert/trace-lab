import type { MdlProblem, Step } from './types';

export interface TraceOptions {
  /** Hard cap on steps, so a pathological problem can't loop forever. */
  maxSteps?: number;
  /**
   * Greedy threshold. A move is only taken if it reduces total bits by more
   * than this. 0 = take any strictly-improving move (pure MDL hill descent).
   * A small positive value models "don't bother unless it really helps".
   */
  minGain?: number;
}

/**
 * The whole search, captured as a list of snapshots.
 *
 * This is the architectural keystone of the app: the algorithm does NOT drive
 * the UI. It runs to completion up front and emits an immutable `Step[]`. The
 * UI is then a pure function of `steps[index]`. Stepping, slow-motion,
 * scrubbing backward, and the evolution charts are all just "pick an index".
 *
 * At each step we:
 *   1. record the current model and its cost,
 *   2. enumerate + score every candidate move (so the user can see the ones
 *      that were rejected, not only the winner),
 *   3. pick the single best strictly-improving move (greedy),
 *   4. apply it and loop.
 *
 * When no move improves the total, MDL has found a local optimum and we stop.
 */
export function trace<Model, Move>(
  problem: MdlProblem<Model, Move>,
  opts: TraceOptions = {}
): Step<Model, Move>[] {
  const maxSteps = opts.maxSteps ?? 200;
  const minGain = opts.minGain ?? 0;

  const steps: Step<Model, Move>[] = [];
  let model = problem.initialModel();

  for (let i = 0; i < maxSteps; i++) {
    const cost = problem.cost(model);
    const moves = problem.candidates(model);
    const scored = moves
      .map((m) => problem.scoreMove(model, m, cost))
      .sort((a, b) => a.delta - b.delta);

    // Greedy: the best move is the most-negative delta, if it clears the bar.
    const best =
      scored.length > 0 && scored[0].delta < -minGain ? scored[0] : null;

    steps.push({
      index: i,
      model,
      cost,
      candidates: scored,
      chosen: best,
      note: best
        ? `Apply ${best.label} — saves ${(-best.delta).toFixed(2)} bits.`
        : scored.length === 0
          ? 'No candidate moves remain. Converged.'
          : 'No move reduces the total. MDL has found its optimum — stop.'
    });

    if (!best) break;
    model = problem.apply(model, best.move);
  }

  return steps;
}
