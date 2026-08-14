<script lang="ts">
  /**
   * The reasoning trace as flowing text: one <button> per generated token,
   * shaded by its surprisal (−log₂ p at emission — bright = the model was
   * unsure, invisible = the token was free). <think>…</think> renders as a
   * visually distinct region: that's the model buying down the answer's cost.
   *
   * `reveal` dims tokens past the player index during playback, so scrubbing
   * replays the trace being born without hiding the surrounding context.
   */
  import type { ReasonTok } from '../../lib/logit/api';

  let {
    steps,
    selected,
    reveal,
    onPick,
    overlay = null,
    prefix = null
  }: {
    steps: ReasonTok[];
    selected: number;
    reveal: number;
    onPick: (i: number) => void;
    /** 0..1 per step: attention received from the selected token's computation.
     * When set, tokens tint by this (purple) instead of surprisal (yellow). */
    overlay?: number[] | null;
    /** The templated prompt, shown dim before the trace; `a` tints by the same
     * attention scale as `overlay` (0 = no tint). */
    prefix?: { t: string; a: number }[] | null;
  } = $props();

  const surp = (s: ReasonTok) => -Math.log2(Math.max(s.p, 1e-30));
  /** 0..1 heat: ~8 bits (p ≈ 0.4%) saturates. */
  const heat = (s: ReasonTok) => Math.min(1, surp(s) / 8);

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
   */
  const OPEN = '<think>';
  const CLOSE = '</think>';

  /** The trace as one string, plus each token's character offset into it. */
  const laid = $derived.by(() => {
    const starts: number[] = [];
    let off = 0;
    for (const s of steps) {
      starts.push(off);
      off += s.t.length;
    }
    return { starts, text: steps.map((s) => s.t).join('') };
  });

  /** Did the TEMPLATE already open a think block before the trace began?
   * R1's chat template ends with '<think>\n', so generation starts mid-thought
   * and the opening marker never appears in `steps` at all. */
  const prefixOpen = $derived.by(() => {
    if (!prefix) return false;
    const p = prefix.map((x) => x.t).join('');
    return p.lastIndexOf(OPEN) > p.lastIndexOf(CLOSE);
  });

  /** Character ranges: `think` = shaded regions, `marks` = the markers alone. */
  const regions = $derived.by(() => {
    const text = laid.text;
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
    return { think, marks };
  });

  const overlaps = (rs: [number, number][], a: number, b: number) =>
    rs.some(([x, y]) => a < y && b > x);
  const span = (i: number): [number, number] => [
    laid.starts[i],
    laid.starts[i] + steps[i].t.length
  ];

  const inThink = $derived(steps.map((_s, i) => overlaps(regions.think, ...span(i))));
  const isMarker = $derived(steps.map((_s, i) => overlaps(regions.marks, ...span(i))));

  /**
   * Split a token for rendering. Buttons are atomic inline-blocks, so a
   * newline INSIDE one becomes a multi-line box whose baseline is its empty
   * last line — hoisting glyphs like the '.' of a merged '.⏎⏎' token two
   * lines above the text. So: glyphs stay in the button, newlines move
   * outside it into the normal flow. A pure-newline token keeps a dim '↵'
   * in the button so it stays clickable.
   */
  const split = (t: string) => {
    const s = t.replaceAll('⏎', '\n');
    const m = /^(\n*)([\s\S]*?)(\n*)$/.exec(s)!;
    const mid = m[2].replaceAll('\n', '↵'); // interior newlines: rare, shown inline
    return { lead: m[1], vis: mid === '' ? '↵' : mid, isBreak: mid === '', trail: m[3] };
  };
</script>

<div class="trace mono scrollbar">
  {#if prefix}{#each prefix as pt, i (i)}{@const q = split(pt.t)}{q.lead}<span
        class="ptok"
        class:patt={pt.a > 0}
        style="--att: {pt.a.toFixed(3)}"
        title={`prompt #${i} “${pt.t}”`}>{q.vis}</span
      >{q.trail}{/each}{/if}
  {#each steps as s, i (s.pos)}{@const p = split(s.t)}{p.lead}<button
      class="tok"
      class:think={inThink[i]}
      class:marker={isMarker[i]}
      class:break={p.isBreak}
      class:sel={i === selected}
      class:unrevealed={i > reveal}
      class:att={overlay !== null}
      style={overlay !== null
        ? `--att: ${(overlay[i] ?? 0).toFixed(3)}`
        : `--heat: ${heat(s).toFixed(3)}`}
      title={`#${s.pos} “${s.t}” — ${surp(s).toFixed(2)}b (p=${(s.p * 100).toFixed(1)}%)`}
      onclick={() => onPick(i)}>{p.vis}</button
    >{p.trail}{/each}
</div>

<style>
  .trace {
    overflow: auto;
    min-height: 0;
    font-size: 13px;
    line-height: 1.85;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .tok {
    all: unset;
    cursor: pointer;
    white-space: pre-wrap;
    border-radius: 3px;
    /* Surprisal heat: transparent when the token was free. */
    background: color-mix(in srgb, var(--data) calc(var(--heat) * 55%), transparent);
    transition: background 0.1s ease;
  }
  .tok:hover {
    outline: 1px solid var(--muted);
  }
  .tok.att {
    /* Attention overlay: where the selected token's computation looked. */
    background: color-mix(in srgb, var(--model) calc(var(--att) * 70%), transparent);
  }
  .ptok {
    color: var(--faint);
    white-space: pre-wrap;
    border-radius: 3px;
  }
  .ptok.patt {
    background: color-mix(in srgb, var(--model) calc(var(--att) * 70%), transparent);
    color: var(--muted);
  }
  .tok.think {
    color: var(--muted);
    font-style: italic;
  }
  .tok.marker {
    color: var(--model);
    font-style: normal;
    font-size: 10.5px;
    letter-spacing: 0.03em;
  }
  .tok.break {
    /* Pure-newline tokens: a faint ↵ so the token is still selectable. */
    color: var(--faint);
    font-size: 10px;
  }
  .tok.sel {
    outline: 2px solid var(--model);
    background: color-mix(in srgb, var(--model) 25%, transparent);
  }
  .tok.unrevealed {
    opacity: 0.22;
  }
</style>
