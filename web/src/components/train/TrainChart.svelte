<script lang="ts">
  /**
   * Loss + held-out exact-match over epochs, streaming in live. Two scales in
   * one frame: loss normalized to its own max (data-colored), accuracy on a
   * fixed 0–1 scale (total-colored) — the crossing point is the fun part.
   */
  let {
    rows
  }: {
    rows: { epoch: number; loss: number; acc: number }[];
  } = $props();

  const W = 520;
  const H = 180;
  const PAD = { l: 6, r: 6, t: 10, b: 16 };

  const maxLoss = $derived(Math.max(0.001, ...rows.map((r) => r.loss)));
  const n = $derived(rows.length);

  const xAt = (i: number) =>
    PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const yLoss = (v: number) => H - PAD.b - (v / maxLoss) * (H - PAD.t - PAD.b);
  const yAcc = (v: number) => H - PAD.b - v * (H - PAD.t - PAD.b);

  const path = (ys: number[]) =>
    ys.map((y, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${y.toFixed(1)}`).join(' ');

  const last = $derived(rows[rows.length - 1]);
</script>

<div class="chart">
  <div class="chead">
    <div class="legend">
      <span class="li"><span class="swatch" style="background:var(--data)"></span>loss (÷max)</span>
      <span class="li"><span class="swatch" style="background:var(--total)"></span>held-out exact-match</span>
    </div>
  </div>

  {#if n > 0}
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="presentation">
      <!-- 50% and 100% accuracy guides — the checkpoint thresholds -->
      <line x1={PAD.l} x2={W - PAD.r} y1={yAcc(0.5)} y2={yAcc(0.5)} class="guide" />
      <line x1={PAD.l} x2={W - PAD.r} y1={yAcc(1)} y2={yAcc(1)} class="guide" />

      <path d={path(rows.map((r) => yLoss(r.loss)))} class="line data" />
      <path d={path(rows.map((r) => yAcc(r.acc)))} class="line total" />

      {#each rows as r, i}
        <circle cx={xAt(i)} cy={yAcc(r.acc)} r={i === n - 1 ? 3.5 : 2} class="dot" />
      {/each}
    </svg>

    <div class="xaxis">
      <span class="faint">epoch 1</span>
      {#if last}
        <span class="cur mono">epoch {last.epoch} · loss {last.loss.toFixed(4)} · {(last.acc * 100).toFixed(1)}%</span>
      {/if}
      <span class="faint">epoch {last?.epoch ?? 1}</span>
    </div>
  {:else}
    <div class="waiting faint mono">no epochs yet — run to start</div>
  {/if}
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; }
  .chead { display: flex; justify-content: flex-end; align-items: center; flex: 0 0 auto; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); }
  .li { display: inline-flex; align-items: center; gap: 4px; }
  svg { width: 100%; flex: 1 1 auto; min-height: 90px; height: auto; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.total { stroke: var(--total); }
  .line.data { stroke: var(--data); opacity: 0.95; }
  .guide { stroke: var(--border-2); stroke-width: 1; stroke-dasharray: 4 4; vector-effect: non-scaling-stroke; }
  .dot { fill: var(--total); }
  .xaxis { display: flex; justify-content: space-between; font-size: 11px; }
  .cur { color: var(--total); }
  .waiting { flex: 1; display: grid; place-items: center; font-size: 11px; }
</style>
