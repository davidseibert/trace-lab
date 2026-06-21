<script lang="ts">
  import type { Step } from '../lib/mdl/types';
  import { fmt } from '../lib/mdl/format';

  let {
    steps,
    index,
    onSeek
  }: {
    steps: Step<unknown, unknown>[];
    index: number;
    onSeek: (i: number) => void;
  } = $props();

  const W = 520;
  const H = 180;
  const PAD = { l: 6, r: 6, t: 10, b: 16 };

  const series = $derived.by(() => {
    const total = steps.map((s) => s.cost.total);
    const model = steps.map((s) => s.cost.modelBits);
    const data = steps.map((s) => s.cost.dataBits);
    const max = Math.max(1, ...total);
    return { total, model, data, max };
  });

  const n = $derived(steps.length);
  const xAt = (i: number) =>
    PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const yAt = (v: number) =>
    H - PAD.b - (v / series.max) * (H - PAD.t - PAD.b);

  const path = (arr: number[]) =>
    arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`).join(' ');

  function handleClick(e: MouseEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const frac = (x - PAD.l) / (W - PAD.l - PAD.r);
    onSeek(Math.round(frac * (n - 1)));
  }
</script>

<div class="chart">
  <div class="chead">
    <span class="panel-title">Description length over search</span>
    <div class="legend">
      <span class="li"><span class="swatch" style="background:var(--total)"></span>total</span>
      <span class="li"><span class="swatch" style="background:var(--model)"></span>L(M)</span>
      <span class="li"><span class="swatch" style="background:var(--data)"></span>L(D|M)</span>
    </div>
  </div>

  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    <!-- current-step marker -->
    <line x1={xAt(index)} x2={xAt(index)} y1={PAD.t} y2={H - PAD.b} class="marker" />

    <path d={path(series.data)} class="line data" />
    <path d={path(series.model)} class="line model" />
    <path d={path(series.total)} class="line total" />

    {#each steps as _, i}
      <circle cx={xAt(i)} cy={yAt(series.total[i])} r={i === index ? 4 : 2.2}
              class="dot" class:active={i === index} />
    {/each}
  </svg>

  <div class="xaxis">
    <span class="faint">step 0</span>
    <span class="cur mono">step {index} · {fmt(series.total[index] ?? 0, 0)} bits</span>
    <span class="faint">step {n - 1}</span>
  </div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; }
  .chead { display: flex; justify-content: space-between; align-items: center; flex: 0 0 auto; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); }
  .li { display: inline-flex; align-items: center; gap: 4px; }
  svg { width: 100%; flex: 1 1 auto; min-height: 90px; height: auto; cursor: crosshair; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.total { stroke: var(--total); }
  .line.model { stroke: var(--model); opacity: 0.95; }
  .line.data { stroke: var(--data); opacity: 0.95; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot { fill: var(--total); }
  .dot.active { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; }
  .cur { color: var(--total); }
</style>
