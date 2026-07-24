/**
 * Generic Minimum Description Length (MDL) types — domain-agnostic.
 *
 * The whole principle in one line:
 *
 *     The best model M of data D is the one that minimizes the TOTAL number
 *     of bits needed to describe the model AND the data given the model:
 *
 *         total = L(M) + L(D | M)
 *                 ^^^^   ^^^^^^^^
 *                 model  data re-encoded using the model
 *
 * A model that is too simple makes L(M) tiny but L(D|M) huge (you must spell
 * everything out). A model that is too complex makes L(D|M) tiny but L(M) huge
 * (you memorized the data as rules). MDL finds the trough between them — and it
 * cannot overfit, because complexity literally costs bits. Parsimony is the
 * objective, not a regularizer bolted on.
 *
 * Everything below is pure data. The string adapter and graph adapter both
 * implement `MdlProblem`, and the same `trace()` runner drives both.
 */

/** One labelled component of a description length, in bits. */
export interface CostTerm {
  /** Human label, e.g. "compressed stream" or "rule bodies". */
  label: string;
  /** Bits this term contributes. */
  bits: number;
  /** Optional one-line explanation of how the bits were computed. */
  detail?: string;
  /** If true, this term is constant across all models (e.g. the base alphabet)
   *  and therefore does not influence which move MDL prefers. Shown for honesty. */
  fixed?: boolean;
}

/** A full L(M) + L(D|M) accounting for one model state. */
export interface CostBreakdown {
  /** L(M): bits to describe the model itself. */
  modelBits: number;
  /** L(D|M): bits to describe the data once the model is known. */
  dataBits: number;
  /** modelBits + dataBits. */
  total: number;
  /** Itemised breakdown of L(M). */
  modelTerms: CostTerm[];
  /** Itemised breakdown of L(D|M). */
  dataTerms: CostTerm[];
  /** Free-form scalars worth surfacing (bits/symbol, vocabulary size, ...). */
  meta: Record<string, number>;
}

/** A scored candidate move: what it is, and what it would cost if applied. */
export interface ScoredMove<Move> {
  move: Move;
  /** Short human description, e.g. "t·h → R0   (×7, expands to “th”)". */
  label: string;
  /** Change in TOTAL bits if applied. Negative = compression gain. */
  delta: number;
  /** Total bits after applying. */
  totalAfter: number;
  /** L(M) after applying. */
  modelBitsAfter: number;
  /** L(D|M) after applying. */
  dataBitsAfter: number;
  /** Extra columns for the candidates table (frequency, gain parts, ...). */
  extra: Record<string, number | string>;
}

/** One snapshot in the search trace — everything the UI needs to render. */
export interface Step<Model, Move> {
  index: number;
  /** Model state AT this step. */
  model: Model;
  /** Cost of this model state. */
  cost: CostBreakdown;
  /** Every candidate move evaluated from this state, sorted best-first. */
  candidates: ScoredMove<Move>[];
  /** The candidate that was applied to reach the next step (null = converged). */
  chosen: ScoredMove<Move> | null;
  /** Narration of what happens at this step. */
  note: string;
}

/**
 * A domain adapter. Implement this for strings, graphs, anything. The runner
 * does the rest: greedy search + full trace capture.
 */
export interface MdlProblem<Model, Move> {
  /** Display name of the lens, e.g. "Grammar compression". */
  name: string;
  /** Short description shown in the UI. */
  blurb: string;
  /** The starting model: no structure learned yet. */
  initialModel(): Model;
  /** Compute the full cost accounting for a model. */
  cost(model: Model): CostBreakdown;
  /** Enumerate all candidate moves available from this model. */
  candidates(model: Model): Move[];
  /** Produce the model that results from applying a move. Must be pure. */
  apply(model: Model, move: Move): Model;
  /** Score a candidate against a baseline cost (usually cost(model)). */
  scoreMove(model: Model, move: Move, baseline: CostBreakdown): ScoredMove<Move>;
}
