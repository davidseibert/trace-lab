<script lang="ts">
  /**
   * Per-token code length across the whole trace: the surprisal profile of the
   * model thinking. Spikes are decision points; long flat valleys are tokens
   * the reasoning already paid for. Click to select a token.
   */
  import type { ReasonTok } from '../../lib/logit/api';
  import { chartScale } from '../../lib/chart';
  import { surprisal, thinkRegions } from '../../lib/reason';

  let {
    steps,
    selected,
    onPick,
    prefixText = null
  }: {
    steps: ReasonTok[];
    selected: number;
    onPick: (i: number) => void;
    /** The templated prompt as one string — lets thinkRegions() see a think
     * block the chat template pre-opened (DeepSeek-R1). */
    prefixText?: string | null;
  } = $props();

  const W = 1000; // viewBox units; the SVG stretches to the panel
  const H = 120;
  const PAD = 4; // slim uniform frame — this strip hugs its panel

  const maxBits = $derived(Math.max(4, ...steps.map(surprisal)));
  const scale = $derived(
    chartScale({ n: steps.length, max: maxBits, W, H, pad: { l: PAD, r: PAD, t: PAD, b: PAD } })
  );
  const path = $derived(steps.length < 2 ? '' : scale.path(steps.map(surprisal)));

  // First think-region span (step indices) for the shaded backdrop.
  const thinkSpan = $derived.by(() => {
    const span = thinkRegions(steps, prefixText).spans[0];
    return span ? { a: span.start, b: span.end } : null;
  });

  function pick(e: MouseEvent) {
    onPick(scale.indexAt(e.clientX, e.currentTarget as Element));
  }
</script>

<div class="wrap">
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={pick} role="img"
    aria-label="per-token code length">
    {#if thinkSpan}
      <rect
        x={scale.xAt(thinkSpan.a)}
        y="0"
        width={scale.xAt(thinkSpan.b) - scale.xAt(thinkSpan.a)}
        height={H}
        class="thinkband"
      />
    {/if}
    {#if path}
      <path d={path} class="curve" />
    {/if}
    {#if steps[selected]}
      <line x1={scale.xAt(selected)} y1="0" x2={scale.xAt(selected)} y2={H} class="cursor" />
      <circle cx={scale.xAt(selected)} cy={scale.yAt(surprisal(steps[selected]))} r="3.5" class="dot" />
    {/if}
  </svg>
  <div class="legend mono faint">
    <span><span class="chip think"></span> think</span>
    <span>peak {maxBits.toFixed(1)}b</span>
    {#if steps[selected]}
      <span>#{steps[selected].pos} “{steps[selected].t}” {surprisal(steps[selected]).toFixed(2)}b</span>
    {/if}
  </div>
</div>

<style>
  .wrap { display: flex; flex-direction: column; gap: 4px; min-height: 0; height: 100%; }
  svg { flex: 1; width: 100%; min-height: 0; cursor: crosshair; }
  .thinkband { fill: color-mix(in srgb, var(--model) 10%, transparent); }
  .curve { fill: none; stroke: var(--data); stroke-width: 1.4; vector-effect: non-scaling-stroke; }
  .cursor { stroke: var(--model); stroke-width: 1; vector-effect: non-scaling-stroke; opacity: 0.8; }
  .dot { fill: var(--model); }
  .legend { display: flex; gap: 14px; font-size: 10.5px; }
  .chip { display: inline-block; width: 9px; height: 9px; border-radius: 2px; vertical-align: -1px; }
  .chip.think { background: color-mix(in srgb, var(--model) 25%, transparent); }
</style>
