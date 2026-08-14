/**
 * CODER LENS — arithmetic coding as the operational back-end every other lens
 * shares. Each lens reduces a model to a per-symbol cost of -log2(p) bits; this
 * lens turns such a probability stream into an actual interval-narrowing
 * encode/decode round-trip, making the fractional bits *visible*.
 *
 * The contract is deliberately model-agnostic: a `CodeStream` is an ordered list
 * of `CodeSymbol`s, each carrying the full normalised distribution AT that step
 * plus which entry was actually emitted. A STATIC model (uniform, empirical,
 * grammar) repeats the same `dist` every step; a CONTEXT model (a future
 * Mini-GPT source) would vary `dist` per step. Same arithmetic core handles both.
 */

import { surprisal } from '../mdl/format';
import {
  token,
  type GrammarModel
} from '../string/grammar';
import { softmaxVec } from '../llm/tensor';
import type { ForwardViz } from '../llm/model';

/** One entry of a next-symbol distribution. Entries of a step sum to p≈1. */
export interface CodeDistEntry {
  /** Stable symbol id (alphabet index / grammar symbol id) — drives colour. */
  id: number;
  /** Display label for the symbol. */
  label: string;
  /** Probability mass. */
  p: number;
}

/** One coded position: the distribution in force, and the symbol emitted. */
export interface CodeSymbol {
  label: string;
  /** Full distribution at this step, in cumulative (display) order. */
  dist: CodeDistEntry[];
  /** Index into `dist` of the symbol actually emitted. */
  chosenIndex: number;
}

export type CodeStream = CodeSymbol[];

/**
 * Float64 intervals stay exact only while the interval width is above ~2⁻⁵²,
 * i.e. while the accumulated bits stay under ~52. We keep a margin and cap the
 * coded prefix so encode/decode round-trip exactly and stay legible.
 */
export const MAX_CHARS = 14;
const MAX_BITS = 48;

/** Render a single character visibly (spaces / newlines made explicit). */
export function showChar(ch: string): string {
  if (ch === ' ') return '␣';
  if (ch === '\n') return '⏎';
  if (ch === '\t') return '⇥';
  return ch;
}

/** Distinct characters of `text`, in first-appearance order. */
function alphabetOf(text: string): { chars: string[]; idOf: Map<string, number> } {
  const chars: string[] = [];
  const idOf = new Map<string, number>();
  for (const ch of text) {
    if (!idOf.has(ch)) {
      idOf.set(ch, chars.length);
      chars.push(ch);
    }
  }
  return { chars, idOf };
}

/** Build a stream over `dist` where every position shares the same `dist`. */
function staticStream(
  text: string,
  dist: CodeDistEntry[],
  idOf: Map<string, number>
): CodeStream {
  const indexOfId = new Map(dist.map((e, i) => [e.id, i]));
  return [...text].map((ch) => {
    const id = idOf.get(ch)!;
    return { label: showChar(ch), dist, chosenIndex: indexOfId.get(id)! };
  });
}

/** Uniform model: every symbol of the alphabet is equally likely (p = 1/V). */
export function uniformStream(text: string): CodeStream {
  const { chars, idOf } = alphabetOf(text);
  const V = chars.length;
  const p = V > 0 ? 1 / V : 1;
  const dist: CodeDistEntry[] = chars.map((ch, id) => ({ id, label: showChar(ch), p }));
  return staticStream(text, dist, idOf);
}

/** Empirical (Shannon) model: p = count/total from the text itself. Expected
 *  cost is Σ p·(−log2 p) = H(text) bits/char — the memoryless lower bound —
 *  versus the uniform model's flat log2(V) ≥ H. The gap between the two streams
 *  is exactly what knowing the letter frequencies buys. */
export function empiricalStream(text: string): CodeStream {
  const { chars, idOf } = alphabetOf(text);
  const counts = new Array(chars.length).fill(0);
  for (const ch of text) counts[idOf.get(ch)!]++;
  const total = text.length || 1;
  const dist: CodeDistEntry[] = chars.map((ch, id) => ({
    id,
    label: showChar(ch),
    p: counts[id] / total
  }));
  return staticStream(text, dist, idOf);
}

/**
 * Grammar source: code the CONVERGED grammar's symbol sequence under an
 * empirical distribution over THAT sequence. The distribution is recomputed and
 * renormalised over the sequence's own alphabet — grammar's `codeLen` is
 * normalised over sequence + rule bodies, so it would not sum to 1 here.
 */
export function grammarStream(model: GrammarModel): CodeStream {
  const counts = new Map<number, number>();
  for (const id of model.sequence) counts.set(id, (counts.get(id) ?? 0) + 1);
  const total = model.sequence.length || 1;
  const ids = [...counts.keys()].sort((a, b) => a - b);
  const dist: CodeDistEntry[] = ids.map((id) => ({
    id,
    label: token(model, id),
    p: counts.get(id)! / total
  }));
  const indexOfId = new Map(ids.map((id, i) => [id, i]));
  return model.sequence.map((id) => ({
    label: token(model, id),
    dist,
    chosenIndex: indexOfId.get(id)!
  }));
}

/**
 * Mini-GPT source: code a token sequence under the model's own next-token
 * predictions. One causal forward pass yields every conditional P(xᵢ₊₁ | x≤ᵢ) at
 * once (teacher forcing), so position i's softmaxed logits ARE the distribution
 * for token i+1. We therefore code tokens 1…T-1 (token 0 is the given seed); the
 * split now CHANGES every step because the model conditions on context.
 */
export function llmStream(viz: ForwardViz, vocab: string[]): CodeStream {
  const [T, V] = viz.logits.shape;
  const data = viz.logits.data;
  const stream: CodeStream = [];

  // The first token has no left context, so the model can't predict it — code it
  // under a uniform prior (the standard choice without a BOS token). It costs a
  // flat log2(V) bits no matter how trained the model is. Every later token uses
  // the model's own next-token distribution, so all sources code the full string.
  const t0 = viz.tokenIds[0];
  if (t0 !== undefined) {
    const uniform: CodeDistEntry[] = [];
    for (let id = 0; id < V; id++) uniform.push({ id, label: vocab[id], p: 1 / V });
    stream.push({ label: vocab[t0], dist: uniform, chosenIndex: t0 });
  }

  for (let i = 0; i < T - 1; i++) {
    const probs = softmaxVec(data.subarray(i * V, (i + 1) * V));
    const dist: CodeDistEntry[] = [];
    for (let id = 0; id < V; id++) dist.push({ id, label: vocab[id], p: probs[id] });
    const target = viz.tokenIds[i + 1];
    stream.push({ label: vocab[target], dist, chosenIndex: target });
  }
  return stream;
}

/**
 * Trim a stream to the longest prefix whose accumulated ideal bits stay under
 * the float-precision budget, so the interval never collapses below resolution.
 */
export function clampStream(stream: CodeStream): { stream: CodeStream; truncated: boolean } {
  let bits = 0;
  for (let i = 0; i < stream.length; i++) {
    const sym = stream[i];
    bits += surprisal(sym.dist[sym.chosenIndex]?.p ?? 1);
    if (bits > MAX_BITS) return { stream: stream.slice(0, i), truncated: true };
  }
  return { stream, truncated: false };
}
