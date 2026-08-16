<script lang="ts">
  /**
   * Energy vs update iteration. Modern energy is charted in bits (nats ÷ ln 2,
   * constants included so E ≥ 0 — the house [0, max] scale applies directly).
   * The classical −½ξᵀWξ is unitless and typically negative, so in `both`
   * mode it rides its own min–max normalized scale; the legend says so.
   */
  import { chartScale } from '../../lib/chart';
  import type { ComboStep } from '../../lib/hopfield/hopfield';

  let {
    steps,
    index,
    onSeek
  }: {
    steps: ComboStep[];
    index: number;
    onSeek: (i: number) => void;
  } = $props();

  const W = 520;
  const H = 180;

  const modern = $derived(steps.map((s) => s.modern?.energyBits ?? null));
  const classical = $derived(steps.map((s) => s.classical?.energy ?? null));
  const hasModern = $derived(modern.some((v) => v !== null));
  const hasClassical = $derived(classical.some((v) => v !== null));

  const modernMax = $derived(Math.max(1e-9, ...modern.filter((v): v is number => v !== null)));
  const scale = $derived(chartScale({ n: steps.length, max: modernMax, W, H }));

  // Classical series normalized to [0, 1] on the same frame (its zero is not
  // meaningful next to the modern bits axis; only its shape is).
  const classicalNorm = $derived.by(() => {
    const vals = classical.filter((v): v is number => v !== null);
    if (!vals.length) return [];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const span = hi - lo || 1;
    return classical.map((v) => (v === null ? null : ((v - lo) / span) * modernMax));
  });

  function pathOf(series: (number | null)[]): string {
    return series
      .map((v, i) => (v === null ? '' : `${i === 0 || series[i - 1] === null ? 'M' : 'L'} ${scale.xAt(i).toFixed(1)} ${scale.yAt(v).toFixed(1)}`))
      .filter(Boolean)
      .join(' ');
  }

  function handleClick(e: MouseEvent) {
    onSeek(scale.indexAt(e.clientX, e.currentTarget as Element));
  }
</script>

<div class="chart">
  <div class="chead">
    <div class="legend">
      {#if hasModern}<span class="li"><span class="swatch" style="background:var(--model)"></span>E modern (bits)</span>{/if}
      {#if hasClassical}<span class="li"><span class="swatch" style="background:var(--data)"></span>E classical (own scale)</span>{/if}
    </div>
  </div>

  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    <line x1={scale.xAt(index)} x2={scale.xAt(index)} y1={scale.top} y2={scale.bottom} class="marker" />

    {#if hasClassical}<path d={pathOf(classicalNorm)} class="line data" />{/if}
    {#if hasModern}<path d={pathOf(modern)} class="line model" />{/if}

    {#each steps as _, i (i)}
      {#if modern[i] !== null}
        <circle cx={scale.xAt(i)} cy={scale.yAt(modern[i]!)} r={i === index ? 4 : 2.2} class="dot model" class:active={i === index} />
      {:else if classicalNorm[i] !== null && classicalNorm[i] !== undefined}
        <circle cx={scale.xAt(i)} cy={scale.yAt(classicalNorm[i]!)} r={i === index ? 4 : 2.2} class="dot data" class:active={i === index} />
      {/if}
    {/each}
  </svg>

  <div class="xaxis">
    <span class="faint">iter 0</span>
    <span class="cur mono">
      iter {index}{#if steps[index]?.modern} · E = {steps[index].modern!.energyBits.toFixed(2)} bits{/if}
    </span>
    <span class="faint">iter {steps.length - 1}</span>
  </div>
  <div class="footnote faint">lse energy ÷ ln 2; the paper’s constants ((1/β)·ln N + ½M²) are included so E ≥ 0.</div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; }
  .chead { display: flex; justify-content: flex-end; align-items: center; flex: 0 0 auto; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); }
  .li { display: inline-flex; align-items: center; gap: 4px; }
  svg { width: 100%; flex: 1 1 auto; min-height: 90px; height: auto; cursor: crosshair; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.model { stroke: var(--model); }
  .line.data { stroke: var(--data); opacity: 0.95; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot.model { fill: var(--model); }
  .dot.data { fill: var(--data); }
  .dot.active { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; }
  .cur { color: var(--model); }
  .footnote { font-size: 10px; }
</style>
