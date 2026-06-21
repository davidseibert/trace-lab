<script lang="ts">
  import type { LlmStep } from '../../lib/llm/trainTrace';

  let {
    steps,
    index,
    onSeek
  }: {
    steps: LlmStep[];
    index: number;
    onSeek: (i: number) => void;
  } = $props();

  const W = 520;
  const H = 180;
  const PAD = { l: 8, r: 8, t: 12, b: 16 };

  const series = $derived(steps.map((s) => s.loss));
  const max = $derived(Math.max(1e-6, ...series));
  const n = $derived(steps.length);

  const xAt = (i: number) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const yAt = (v: number) => H - PAD.b - (v / max) * (H - PAD.t - PAD.b);

  const path = $derived(
    series.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`).join(' ')
  );

  function handleClick(e: MouseEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const frac = (x - PAD.l) / (W - PAD.l - PAD.r);
    onSeek(Math.round(frac * (n - 1)));
  }
</script>

<div class="chart">
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    <line x1={xAt(index)} x2={xAt(index)} y1={PAD.t} y2={H - PAD.b} class="marker" />
    <path d={path} class="line" />
    <circle cx={xAt(index)} cy={yAt(series[index] ?? 0)} r="4" class="dot" />
  </svg>

  <div class="xaxis">
    <span class="faint">step 0</span>
    <span class="cur mono">step {index} · loss {(series[index] ?? 0).toFixed(3)}</span>
    <span class="faint">step {n - 1}</span>
  </div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; flex: 1 1 auto; }
  svg { width: 100%; flex: 1 1 auto; min-height: 80px; height: auto; cursor: crosshair; display: block; }
  .line { fill: none; stroke: var(--data); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; flex: 0 0 auto; }
  .cur { color: var(--data); }
</style>
