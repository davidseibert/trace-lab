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
    onPick
  }: {
    steps: ReasonTok[];
    selected: number;
    reveal: number;
    onPick: (i: number) => void;
  } = $props();

  const surp = (s: ReasonTok) => -Math.log2(Math.max(s.p, 1e-30));
  /** 0..1 heat: ~8 bits (p ≈ 0.4%) saturates. */
  const heat = (s: ReasonTok) => Math.min(1, surp(s) / 8);

  // Think-region membership per token, by matching Qwen3's dedicated tokens.
  const inThink = $derived.by(() => {
    let open = false;
    return steps.map((s) => {
      if (s.t === '<think>') open = true;
      const v = open;
      if (s.t === '</think>') open = false;
      return v;
    });
  });

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
  {#each steps as s, i (s.pos)}{@const p = split(s.t)}{p.lead}<button
      class="tok"
      class:think={inThink[i]}
      class:marker={s.t === '<think>' || s.t === '</think>'}
      class:break={p.isBreak}
      class:sel={i === selected}
      class:unrevealed={i > reveal}
      style="--heat: {heat(s).toFixed(3)}"
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
