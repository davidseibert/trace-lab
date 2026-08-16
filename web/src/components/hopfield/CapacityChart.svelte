<script lang="ts">
  /**
   * Retrieval success rate vs number of stored patterns, modern and classical
   * on the same patterns and corruptions. The dashed red line marks the
   * classical Hebbian capacity ~0.138·d — the wall the lse energy removes.
   */
  import { chartScale } from '../../lib/chart';
  import type { CapacityPoint } from '../../lib/hopfield/hopfield';

  let {
    points,
    d
  }: {
    points: CapacityPoint[];
    /** Pattern dimension the sweep ran at (for the 0.138·d annotation). */
    d: number;
  } = $props();

  const W = 520;
  const H = 180;

  const scale = $derived(chartScale({ n: points.length, max: 1, W, H }));

  const modern = $derived(points.map((p) => p.modern));
  const classical = $derived(points.map((p) => p.classical));

  // 0.138·d sits between two sampled Ns — interpolate its x fractionally.
  const hebbX = $derived.by(() => {
    const cap = 0.138 * d;
    const Ns = points.map((p) => p.N);
    if (!Ns.length || cap <= Ns[0] || cap >= Ns[Ns.length - 1]) return null;
    for (let i = 1; i < Ns.length; i++)
      if (cap <= Ns[i]) {
        const f = (cap - Ns[i - 1]) / (Ns[i] - Ns[i - 1]);
        return scale.xAt(i - 1) + f * (scale.xAt(i) - scale.xAt(i - 1));
      }
    return null;
  });
</script>

<div class="chart">
  <div class="chead">
    <div class="legend">
      <span class="li"><span class="swatch" style="background:var(--model)"></span>modern</span>
      <span class="li"><span class="swatch" style="background:var(--data)"></span>classical</span>
      {#if hebbX !== null}<span class="li"><span class="swatch dashed"></span>0.138·d ≈ {(0.138 * d).toFixed(1)}</span>{/if}
    </div>
  </div>

  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="presentation">
    {#if hebbX !== null}
      <line x1={hebbX} x2={hebbX} y1={scale.top} y2={scale.bottom} class="hebb" />
    {/if}
    <path d={scale.path(classical)} class="line data" />
    <path d={scale.path(modern)} class="line model" />
    {#each points as p, i (p.N)}
      <circle cx={scale.xAt(i)} cy={scale.yAt(modern[i])} r="2.6" class="dot model" />
      <circle cx={scale.xAt(i)} cy={scale.yAt(classical[i])} r="2.6" class="dot data" />
    {/each}
  </svg>

  <div class="xaxis mono">
    {#each points as p (p.N)}<span class="faint">{p.N}</span>{/each}
  </div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; }
  .chead { display: flex; justify-content: flex-end; align-items: center; flex: 0 0 auto; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); }
  .li { display: inline-flex; align-items: center; gap: 4px; }
  .swatch.dashed { background: transparent; border-top: 2px dashed var(--bad); width: 12px; height: 0; }
  svg { width: 100%; flex: 1 1 auto; min-height: 90px; height: auto; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.model { stroke: var(--model); }
  .line.data { stroke: var(--data); opacity: 0.95; }
  .hebb { stroke: var(--bad); stroke-width: 1; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }
  .dot.model { fill: var(--model); }
  .dot.data { fill: var(--data); }
  .xaxis { display: flex; justify-content: space-between; font-size: 10px; }
</style>
