/**
 * Shared helpers for the reasoning lens: per-token surprisal, and where the
 * <think>…</think> region(s) of a trace actually are. One implementation so
 * TraceView (shading), TokenBitsStrip (backdrop band) and ReasonApp (Δbits
 * ablation) can never disagree about what counts as "thinking".
 */

/** −log₂ p of a token at emission — its ideal code length in bits. */
export const surprisal = (s: { p: number }) => -Math.log2(Math.max(s.p, 1e-30));

const OPEN = '<think>';
const CLOSE = '</think>';

export interface ThinkInfo {
  /** Per step: does any part of this token overlap a think region?
   * The marker tokens themselves count as in-think. */
  inThink: boolean[];
  /** Per step: does any part of this token overlap a marker itself? */
  isMarker: boolean[];
  /** Think regions as inclusive step-index ranges (markers included),
   * in trace order. Usually 0 or 1 of these. */
  spans: { start: number; end: number }[];
}

/**
 * Think-region membership per token.
 *
 * Token IDENTITY can't be the test. Qwen3 and DeepSeek-R1-Distill both have
 * dedicated marker tokens (151667/8 and 151648/9), so `s.t === '<think>'`
 * looks like it should work — but R1's chat template PRE-OPENS <think> at the
 * end of the prompt. The opening marker is therefore never generated, the
 * flag never flips, and a 220-token reasoning trace renders as entirely
 * un-thought (measured: 0 tokens shaded before this change, 89 after).
 *
 * So work in TEXT space: concatenate the trace, find the markers in the
 * string, and seed the state from the prompt. This also stops caring whether
 * a model spells the markers as one token or several, which is not
 * guaranteed across reasoning models. A token counts as in-think if any part
 * of it overlaps an open region; the markers themselves are included,
 * matching the old behaviour.
 *
 * @param steps the generated trace (only `t` is read)
 * @param prefixText the templated prompt as one string, if the caller has it —
 *   this is what detects R1's pre-opened block. `null` assumes no open block.
 */
export function thinkRegions(steps: { t: string }[], prefixText: string | null = null): ThinkInfo {
  // The trace as one string, plus each token's character offset into it.
  const starts: number[] = [];
  let off = 0;
  for (const s of steps) {
    starts.push(off);
    off += s.t.length;
  }
  const text = steps.map((s) => s.t).join('');

  // Did the TEMPLATE already open a think block before the trace began?
  // R1's chat template ends with '<think>\n', so generation starts mid-thought
  // and the opening marker never appears in `steps` at all.
  const prefixOpen =
    prefixText !== null && prefixText.lastIndexOf(OPEN) > prefixText.lastIndexOf(CLOSE);

  // Character ranges: `think` = whole regions, `marks` = the markers alone.
  const think: [number, number][] = [];
  const marks: [number, number][] = [];
  let open = prefixOpen;
  let start = open ? 0 : -1;
  let i = 0;
  while (i < text.length) {
    if (open) {
      const c = text.indexOf(CLOSE, i);
      if (c === -1) break; // still thinking when the trace ends
      marks.push([c, c + CLOSE.length]);
      think.push([start, c + CLOSE.length]);
      open = false;
      i = c + CLOSE.length;
    } else {
      const o = text.indexOf(OPEN, i);
      if (o === -1) break; // no further think blocks
      marks.push([o, o + OPEN.length]);
      start = o;
      open = true;
      i = o + OPEN.length;
    }
  }
  if (open) think.push([start, text.length]);

  // Back from character ranges to token space.
  const overlaps = (rs: [number, number][], a: number, b: number) =>
    rs.some(([x, y]) => a < y && b > x);
  const span = (j: number): [number, number] => [starts[j], starts[j] + steps[j].t.length];
  const inThink = steps.map((_s, j) => overlaps(think, ...span(j)));
  const isMarker = steps.map((_s, j) => overlaps(marks, ...span(j)));

  const spans: { start: number; end: number }[] = [];
  for (const [a, b] of think) {
    let s0 = -1;
    let s1 = -1;
    for (let j = 0; j < steps.length; j++) {
      const [x, y] = span(j);
      if (x < b && y > a) {
        if (s0 < 0) s0 = j;
        s1 = j;
      }
    }
    if (s0 >= 0) spans.push({ start: s0, end: s1 });
  }

  return { inThink, isMarker, spans };
}
