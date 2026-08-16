<script lang="ts">
  /**
   * Attention mass mapped back onto the board: each key position of the move
   * sequence corresponds to the cell that was played there, so a query's
   * attention row over positions re-renders as heat over cells — "which
   * squares was the model looking at". The start token '·' has no cell; its
   * mass gets a side chip so nothing is silently dropped.
   */
  let {
    mass,
    startMass,
    cellPx = 34
  }: {
    /** Attention mass summed per cell (length 9). */
    mass: Float64Array;
    /** Mass on the '·' start token. */
    startMass: number;
    cellPx?: number;
  } = $props();

  const mx = $derived(Math.max(1e-9, startMass, ...mass));
</script>

<div class="wrap">
  <div class="hgrid" style="grid-template-columns: repeat(3, {cellPx}px); grid-auto-rows: {cellPx}px">
    {#each mass as m, i (i)}
      <div class="hcell" style="background: rgba(255,209,102,{((0.9 * m) / mx).toFixed(3)})" title={`cell ${i}: ${(m * 100).toFixed(1)}% of attention`}>
        {#if m > 0.005}<span class="mono pct">{(m * 100).toFixed(0)}</span>{/if}
      </div>
    {/each}
  </div>
  <div class="start chip mono" title="mass on the '·' start token — position 0 has no cell">
    · {(startMass * 100).toFixed(0)}%
  </div>
</div>

<style>
  .wrap { display: flex; align-items: center; gap: 10px; }
  .hgrid { display: grid; gap: 2px; background: var(--border-2); padding: 2px; border-radius: 5px; width: fit-content; }
  .hcell { display: flex; align-items: center; justify-content: center; background: var(--bg-2); border-radius: 2px; }
  .pct { font-size: 9px; color: var(--text); }
  .start { font-size: 11px; padding: 3px 8px; border: 1px solid var(--border-2); border-radius: 5px; background: var(--bg-2); color: var(--muted); }
</style>
