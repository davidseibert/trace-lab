<script lang="ts">
  /**
   * The paper's phase structure, scrubbable: run the full retrieval at each of
   * 33 log-spaced β and plot the final step's max weight and (normalized)
   * entropy. The background tint IS the regime classification; the labelled
   * tick at β* = 1/√d marks the value attention uses. Clicking sets β.
   */
  import { chartScale } from '../../lib/chart';
  import type { BetaSweepPoint, Regime } from '../../lib/hopfield/hopfield';

  let {
    points,
    betaStar,
    currentBeta,
    N,
    onPick
  }: {
    points: BetaSweepPoint[];
    betaStar: number;
    currentBeta: number;
    N: number;
    onPick: (beta: number) => void;
  } = $props();

  const W = 520;
  const H = 180;

  const scale = $derived(chartScale({ n: points.length, max: 1, W, H }));

  const entNorm = $derived(points.map((p) => p.entropyBits / Math.max(1e-9, Math.log2(Math.max(2, N)))));
  const maxW = $derived(points.map((p) => p.maxWeight));

  const REGIME_TINT: Record<Regime, string> = {
    retrieval: 'rgba(87,217,163,0.10)',
    metastable: 'rgba(255,180,84,0.10)',
    global: 'rgba(255,107,107,0.10)'
  };

  // Nearest sample index for a β value (the grid is log-spaced).
  function idxOf(beta: number): number {
    let best = 0;
    let bd = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(Math.log(p.beta) - Math.log(beta));
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }
  const curIdx = $derived(idxOf(currentBeta));
  const starIdx = $derived(idxOf(betaStar));

  function handleClick(e: MouseEvent) {
    const i = scale.indexAt(e.clientX, e.currentTarget as Element);
    onPick(points[i].beta);
  }

  const bandW = $derived(points.length > 1 ? scale.xAt(1) - scale.xAt(0) : 0);
</script>

<div class="chart">
  <div class="chead">
    <div class="legend">
      <span class="li"><span class="swatch" style="background:var(--model)"></span>max weight</span>
      <span class="li"><span class="swatch" style="background:var(--data)"></span>entropy / log₂N</span>
    </div>
  </div>

  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    {#each points as p, i (i)}
      <rect x={scale.xAt(i) - bandW / 2} y={scale.top} width={bandW} height={scale.bottom - scale.top} fill={REGIME_TINT[p.regime]} />
    {/each}

    <!-- β* = 1/√d: the value attention uses -->
    <line x1={scale.xAt(starIdx)} x2={scale.xAt(starIdx)} y1={scale.top} y2={scale.bottom} class="star" />

    <path d={scale.path(entNorm)} class="line data" />
    <path d={scale.path(maxW)} class="line model" />

    <line x1={scale.xAt(curIdx)} x2={scale.xAt(curIdx)} y1={scale.top} y2={scale.bottom} class="marker" />
    <circle cx={scale.xAt(curIdx)} cy={scale.yAt(maxW[curIdx])} r="4" class="dot active" />
  </svg>

  <div class="xaxis">
    <span class="faint mono">β = {points[0]?.beta.toPrecision(2)}</span>
    <span class="cur mono">β = {currentBeta.toPrecision(3)} ({(currentBeta / betaStar).toPrecision(2)}×β*) · {points[curIdx]?.regime}</span>
    <span class="faint mono">β = {points[points.length - 1]?.beta.toPrecision(2)}</span>
  </div>
  <div class="footnote faint">dashed line: β* = 1/√d — attention’s scaling. Green tint = retrieval, orange = metastable, red = global. Click to set β.</div>
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
  .star { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }
  .dot.active { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; }
  .cur { color: var(--model); }
  .footnote { font-size: 10px; }
</style>
