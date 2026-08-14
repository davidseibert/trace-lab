<script lang="ts">
  /**
   * Per-token code length across the whole trace: the surprisal profile of the
   * model thinking. Spikes are decision points; long flat valleys are tokens
   * the reasoning already paid for. Click to select a token.
   */
  import type { ReasonTok } from '../../lib/logit/api';
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
  const PAD = 4;

  const maxBits = $derived(Math.max(4, ...steps.map(surprisal)));
  const x = (i: number) => PAD + (i / Math.max(1, steps.length - 1)) * (W - 2 * PAD);
  const y = (b: number) => H - PAD - (Math.min(b, maxBits) / maxBits) * (H - 2 * PAD);
  const path = $derived(
    steps.length < 2 ? '' : steps.map((s, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(surprisal(s)).toFixed(1)}`).join(' ')
  );

  // First think-region span (step indices) for the shaded backdrop.
  const thinkSpan = $derived.by(() => {
    const span = thinkRegions(steps, prefixText).spans[0];
    return span ? { a: span.start, b: span.end } : null;
  });

  function pick(e: MouseEvent) {
    const el = e.currentTarget as SVGSVGElement;
    const r = el.getBoundingClientRect();
    const fx = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((fx - PAD) / (W - 2 * PAD)) * (steps.length - 1));
    onPick(Math.max(0, Math.min(steps.length - 1, i)));
  }
</script>

<div class="wrap">
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={pick} role="img"
    aria-label="per-token code length">
    {#if thinkSpan}
      <rect
        x={x(thinkSpan.a)}
        y="0"
        width={x(thinkSpan.b) - x(thinkSpan.a)}
        height={H}
        class="thinkband"
      />
    {/if}
    {#if path}
      <path d={path} class="curve" />
    {/if}
    {#if steps[selected]}
      <line x1={x(selected)} y1="0" x2={x(selected)} y2={H} class="cursor" />
      <circle cx={x(selected)} cy={y(surprisal(steps[selected]))} r="3.5" class="dot" />
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
