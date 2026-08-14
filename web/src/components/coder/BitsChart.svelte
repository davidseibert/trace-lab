<script lang="ts">
  import type { CoderStep } from '../../lib/coder/arithmetic';
  import { chartScale } from '../../lib/chart';

  let {
    steps,
    index,
    refBits,
    onSeek
  }: {
    steps: CoderStep[];
    index: number;
    refBits: number;
    onSeek: (i: number) => void;
  } = $props();

  const W = 520;
  const H = 160;

  const series = $derived(steps.map((s) => s.bitsSoFar));
  const max = $derived(Math.max(1e-6, refBits, ...series));
  const n = $derived(steps.length);
  const scale = $derived(chartScale({ n, max, W, H }));

  const path = $derived(scale.path(series));

  function handleClick(e: MouseEvent) {
    onSeek(scale.indexAt(e.clientX, e.currentTarget as Element));
  }
</script>

<div class="chart">
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    <line x1={scale.left} x2={scale.right} y1={scale.yAt(refBits)} y2={scale.yAt(refBits)} class="ref" />
    <line x1={scale.xAt(index)} x2={scale.xAt(index)} y1={scale.top} y2={scale.bottom} class="marker" />
    <path d={path} class="line" />
    <circle cx={scale.xAt(index)} cy={scale.yAt(series[index] ?? 0)} r="4" class="dot" />
  </svg>

  <div class="xaxis">
    <span class="faint">accumulating bits</span>
    <span class="cur mono">{(series[index] ?? 0).toFixed(2)} bits · codeword {refBits.toFixed(0)}</span>
  </div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; flex: 1 1 auto; }
  svg { width: 100%; flex: 1 1 auto; min-height: 80px; height: auto; cursor: crosshair; display: block; }
  .line { fill: none; stroke: var(--data); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .ref { stroke: var(--total); stroke-width: 1; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; opacity: 0.7; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; flex: 0 0 auto; }
  .cur { color: var(--data); }
</style>
