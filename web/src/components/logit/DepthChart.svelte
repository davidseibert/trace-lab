<script lang="ts">
  // The evolution chart, with depth instead of search steps on the x-axis:
  // −log₂ p(the model's final prediction) at every rung, classic lens vs
  // J-lens. "The prediction sharpens with depth" = this curve falling.
  import { chartScale } from '../../lib/chart';

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

  const n = $derived(bits.length);
  const max = $derived(Math.max(1, uniform * 1.15, ...bits, ...(jbits ?? [])));
  const scale = $derived(chartScale({ n, max, W, H }));

  function handleClick(e: MouseEvent) {
    onSeek(scale.indexAt(e.clientX, e.currentTarget as Element));
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
    <line x1={scale.left} x2={scale.right} y1={scale.yAt(uniform)} y2={scale.yAt(uniform)}
          stroke="var(--muted)" stroke-dasharray="3 4" stroke-width="1" opacity="0.6" />
    {#if jbits}
      <path d={scale.path(jbits)} fill="none" stroke="var(--model)" stroke-width="1.6" />
    {/if}
    <path d={scale.path(bits)} fill="none" stroke="var(--data)" stroke-width="1.8" />

    <line x1={scale.xAt(index)} x2={scale.xAt(index)} y1={scale.top} y2={scale.bottom}
          stroke="var(--text)" stroke-width="1" opacity="0.5" />
    <circle cx={scale.xAt(index)} cy={scale.yAt(bits[index] ?? 0)} r="3.2" fill="var(--data)" />
    {#if jbits}
      <circle cx={scale.xAt(index)} cy={scale.yAt(jbits[index] ?? 0)} r="3.2" fill="var(--model)" />
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
