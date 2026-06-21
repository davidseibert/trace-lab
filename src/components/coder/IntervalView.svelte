<script lang="ts">
  import type { CoderStep } from '../../lib/coder/arithmetic';
  import { symColorFor } from '../../lib/coder/colors';

  let {
    steps,
    index,
    memoryless = true
  }: {
    steps: CoderStep[];
    index: number;
    /** True when every level shares the same split (static model). */
    memoryless?: boolean;
  } = $props();

  // The descent SO FAR: one row per step up to where we're scrubbed. Each row is
  // the chosen slice of the row above, rescaled to full width — the russian doll.
  const rows = $derived(steps.slice(0, index + 1));
  const last = $derived(steps[index]);

  function bandsOf(s: CoderStep) {
    let cum = 0;
    return s.dist.map((e, i) => {
      const lo = cum;
      cum += e.p;
      return { e, i, lo, hi: cum, chosen: i === s.chosenIndex };
    });
  }

  // Where the codeword sits within a given level's interval (decode only).
  function needlePct(s: CoderStep): number | null {
    if (s.value === undefined) return null;
    const t = (s.value - s.lo) / (s.hi - s.lo);
    return Math.max(0, Math.min(1, t)) * 100;
  }

  // Auto-scroll to the newest (deepest) row as playback advances.
  let scroller = $state<HTMLDivElement | null>(null);
  $effect(() => {
    void index;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  });
</script>

<div class="iv">
  <div class="cap">
    <span class="faint">descent — each bar is the one above, zoomed into the chosen slice</span>
    {#if last}<span class="mono faint">[{last.lo.toFixed(5)}, {last.hi.toFixed(5)})</span>{/if}
  </div>

  <div class="stack scrollbar" bind:this={scroller}>
    {#each rows as s, r (r)}
      {@const nx = needlePct(s)}
      <div class="zrow">
        <span class="ridx mono faint">{r}</span>
        <div class="bands">
          {#each bandsOf(s) as b (b.e.id)}
            {@const c = symColorFor(b.e.id)}
            <div
              class="band"
              class:chosen={b.chosen}
              style="flex:{b.e.p}; background:{c.bg}; color:{c.fg}; border-color:{c.border}"
              title="{b.e.label}  p={b.e.p.toFixed(3)}"
            >
              {#if b.e.p > 0.08}<span class="blab mono">{b.e.label}</span>{/if}
            </div>
          {/each}
          {#if nx !== null}<div class="needle" style="left:{nx}%"></div>{/if}
        </div>
      </div>

      {#if r < rows.length - 1}
        {@const cxLo = (s.cumLo * 100).toFixed(2)}
        {@const cxHi = (s.cumHi * 100).toFixed(2)}
        <svg class="funnel" viewBox="0 0 100 14" preserveAspectRatio="none" role="presentation">
          <path d="M {cxLo} 0 L {cxHi} 0 L 100 14 L 0 14 Z" />
        </svg>
      {/if}
    {/each}
  </div>

  {#if last}
    <div class="legend">
      {#if last.phase === 'encode'}
        <span>Fold <b class="mono">{last.chosenLabel}</b> — keep its slice
          <span class="mono">[{last.cumLo.toFixed(3)}, {last.cumHi.toFixed(3)})</span>;
          that becomes the next bar.
          {memoryless
            ? 'Same split each level — the model is memoryless.'
            : 'The split changes each level — the model conditions on context.'}</span>
      {:else}
        <span>The codeword (dashed) lands in <b class="mono">{last.chosenLabel}</b> at every level;
          emit it, zoom in, repeat.</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .iv { display: flex; flex-direction: column; gap: 6px; flex: 1 1 auto; min-height: 0; }
  .cap { display: flex; justify-content: space-between; align-items: baseline; font-size: 11px; }

  .stack { display: flex; flex-direction: column; overflow: auto; min-height: 0; flex: 1 1 auto; }

  .zrow { display: flex; align-items: center; gap: 6px; flex: 0 0 auto; }
  .ridx { font-size: 9px; width: 12px; text-align: right; flex: 0 0 auto; }
  .bands {
    position: relative;
    flex: 1 1 auto;
    display: flex;
    height: 24px;
    border-radius: var(--r-sm);
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .band {
    display: flex; align-items: center; justify-content: center;
    min-width: 1px; opacity: 0.32;
    border-right: 1px solid rgba(0, 0, 0, 0.25);
    transition: opacity 0.2s ease;
  }
  .band.chosen { opacity: 1; outline: 2px solid; outline-offset: -2px; }
  .blab { font-size: 10px; }
  .needle {
    position: absolute; top: -1px; bottom: -1px; width: 0;
    border-left: 1.5px dashed var(--chosen);
    pointer-events: none;
  }

  .funnel { width: 100%; height: 14px; display: block; flex: 0 0 auto; margin-left: 18px; }
  .funnel path { fill: var(--border-2); opacity: 0.4; }

  .legend { font-size: 12px; color: var(--muted); flex: 0 0 auto; }
  .legend b { color: var(--text); }
</style>
