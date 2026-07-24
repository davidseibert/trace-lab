<script lang="ts">
  // The evolution chart, with depth instead of search steps on the x-axis:
  // −log₂ p(the model's final prediction) at every rung, classic lens vs
  // J-lens. "The prediction sharpens with depth" = this curve falling.
  let {
    bits,
    jbits,
    uniform,
    index,
    predToken,
    onSeek
  }: {
    bits: number[];
    jbits: number[] | null;
    uniform: number;
    index: number;
    predToken: string;
    onSeek: (i: number) => void;
  } = $props();

  const W = 520;
  const H = 180;
  const PAD = { l: 6, r: 6, t: 10, b: 16 };

  const n = $derived(bits.length);
  const max = $derived(Math.max(1, uniform * 1.15, ...bits, ...(jbits ?? [])));

  const xAt = (i: number) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const yAt = (v: number) => H - PAD.b - (Math.min(v, max) / max) * (H - PAD.t - PAD.b);

  const path = (arr: number[]) =>
    arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`).join(' ');

  function handleClick(e: MouseEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const frac = (x - PAD.l) / (W - PAD.l - PAD.r);
    onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (n - 1)));
  }
</script>

<div class="chart">
  <div class="chead">
    <div class="legend mono">
      <span class="li"><span class="swatch" style="background:var(--data)"></span>logit lens</span>
      {#if jbits}
        <span class="li"><span class="swatch" style="background:var(--model)"></span>J-lens</span>
      {/if}
      <span class="li faint">— uniform log₂V</span>
    </div>
    <span class="mono faint">−log₂ p({predToken})</span>
  </div>

  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
  <svg viewBox="0 0 {W} {H}" onclick={handleClick} role="img" aria-label="code length by depth">
    <line x1={PAD.l} x2={W - PAD.r} y1={yAt(uniform)} y2={yAt(uniform)}
          stroke="var(--muted)" stroke-dasharray="3 4" stroke-width="1" opacity="0.6" />
    {#if jbits}
      <path d={path(jbits)} fill="none" stroke="var(--model)" stroke-width="1.6" />
    {/if}
    <path d={path(bits)} fill="none" stroke="var(--data)" stroke-width="1.8" />

    <line x1={xAt(index)} x2={xAt(index)} y1={PAD.t} y2={H - PAD.b}
          stroke="var(--text)" stroke-width="1" opacity="0.5" />
    <circle cx={xAt(index)} cy={yAt(bits[index] ?? 0)} r="3.2" fill="var(--data)" />
    {#if jbits}
      <circle cx={xAt(index)} cy={yAt(jbits[index] ?? 0)} r="3.2" fill="var(--model)" />
    {/if}
  </svg>
</div>

<style>
  .chart { display: flex; flex-direction: column; gap: 4px; min-height: 0; flex: 1 1 auto; }
  .chead { display: flex; align-items: center; justify-content: space-between; font-size: 10.5px; }
  .legend { display: flex; gap: 10px; }
  .li { display: inline-flex; align-items: center; gap: 4px; color: var(--muted); }
  .swatch { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
  svg { width: 100%; height: auto; flex: 1 1 auto; min-height: 0; cursor: crosshair; }
</style>
