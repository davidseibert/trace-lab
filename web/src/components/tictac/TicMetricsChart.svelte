<script lang="ts">
  /**
   * The training story in five curves: loss (bits, own scale), and four
   * fractions on a shared 0–1 scale — optimal-move agreement, D₄ equivariance
   * error, weight sparsity, and illegal probability mass. TrainChart's
   * multi-series frame merged with LossChart's click-to-seek.
   */
  import { chartScale } from '../../lib/chart';
  import type { MetricsPoint } from '../../lib/tictac/metrics';

  let {
    points,
    step,
    onSeek
  }: {
    points: MetricsPoint[];
    /** Current training step (the Player's index). */
    step: number;
    onSeek: (step: number) => void;
  } = $props();

  const W = 520;
  const H = 180;

  const scale = $derived(chartScale({ n: points.length, max: 1, W, H }));
  const lossMax = $derived(Math.max(1e-9, ...points.map((p) => p.loss)));

  const lossN = $derived(points.map((p) => p.loss / lossMax));
  const agree = $derived(points.map((p) => p.agreement));
  const equiv = $derived(points.map((p) => p.equivariance));
  const sparse = $derived(points.map((p) => p.sparsity));
  const illegal = $derived(points.map((p) => p.illegalMass));

  // Nearest metric point at or below the scrubbed training step.
  const curIdx = $derived.by(() => {
    let best = 0;
    points.forEach((p, i) => {
      if (p.step <= step) best = i;
    });
    return best;
  });
  const cur = $derived(points[curIdx]);

  function handleClick(e: MouseEvent) {
    const i = scale.indexAt(e.clientX, e.currentTarget as Element);
    onSeek(points[i].step);
  }
</script>

<div class="chart">
  <div class="chead">
    <div class="legend">
      <span class="li"><span class="swatch" style="background:var(--data)"></span>loss (÷{lossMax.toFixed(1)}b)</span>
      <span class="li"><span class="swatch" style="background:var(--good)"></span>agree</span>
      <span class="li"><span class="swatch" style="background:var(--bad)"></span>equiv err</span>
      <span class="li"><span class="swatch" style="background:#8b93a7"></span>sparsity</span>
      <span class="li"><span class="swatch thin" style="background:var(--model)"></span>illegal</span>
    </div>
  </div>

  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" onclick={handleClick} role="presentation">
    <line x1={scale.xAt(curIdx)} x2={scale.xAt(curIdx)} y1={scale.top} y2={scale.bottom} class="marker" />
    <line x1={scale.left} x2={scale.right} y1={scale.yAt(0.5)} y2={scale.yAt(0.5)} class="guide" />

    <path d={scale.path(illegal)} class="line illegal" />
    <path d={scale.path(sparse)} class="line sparse" />
    <path d={scale.path(equiv)} class="line equiv" />
    <path d={scale.path(lossN)} class="line loss" />
    <path d={scale.path(agree)} class="line agree" />

    <circle cx={scale.xAt(curIdx)} cy={scale.yAt(agree[curIdx])} r="4" class="dot" />
  </svg>

  <div class="xaxis">
    <span class="faint mono">step 0</span>
    {#if cur}
      <span class="cur mono">
        step {cur.step} · {cur.loss.toFixed(2)}b · agree {(cur.agreement * 100).toFixed(0)}% · equiv {cur.equivariance.toFixed(2)} · illegal {(cur.illegalMass * 100).toFixed(0)}%
      </span>
    {/if}
    <span class="faint mono">step {points[points.length - 1]?.step ?? 0}</span>
  </div>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; }
  .chead { display: flex; justify-content: flex-end; flex: 0 0 auto; }
  .legend { display: flex; gap: 10px; font-size: 10.5px; color: var(--muted); flex-wrap: wrap; }
  .li { display: inline-flex; align-items: center; gap: 4px; }
  .swatch.thin { height: 1px; }
  svg { width: 100%; flex: 1 1 auto; min-height: 90px; height: auto; cursor: crosshair; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.loss { stroke: var(--data); }
  .line.agree { stroke: var(--good); }
  .line.equiv { stroke: var(--bad); }
  .line.sparse { stroke: #8b93a7; opacity: 0.8; }
  .line.illegal { stroke: var(--model); stroke-width: 1; opacity: 0.8; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .guide { stroke: var(--border-2); stroke-width: 0.5; stroke-dasharray: 4 4; vector-effect: non-scaling-stroke; }
  .dot { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 10.5px; gap: 8px; }
  .cur { color: var(--good); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
