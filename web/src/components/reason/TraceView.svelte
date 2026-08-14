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
  import { surprisal, thinkRegions } from '../../lib/reason';

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

  /** 0..1 heat: ~8 bits (p ≈ 0.4%) saturates. */
  const heat = (s: ReasonTok) => Math.min(1, surprisal(s) / 8);

  // Think-region membership per token — see thinkRegions() for why this is a
  // text-space scan seeded from the prompt, not a token-identity test.
  const think = $derived(thinkRegions(steps, prefix ? prefix.map((x) => x.t).join('') : null));
  const inThink = $derived(think.inThink);
  const isMarker = $derived(think.isMarker);

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
      title={`#${s.pos} “${s.t}” — ${surprisal(s).toFixed(2)}b (p=${(s.p * 100).toFixed(1)}%)`}
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
