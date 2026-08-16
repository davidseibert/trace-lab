<script lang="ts">
  /**
   * One pattern (or state) as a small pixel grid — a flat vector reshaped to
   * rows×cols and pushed through the house diverging colormap, so ±1 patterns
   * read as ink-on-paper and continuous blends read as literal gray ghosts.
   * Fixed max=1 normalization (not per-matrix max) so two grids side by side
   * are directly comparable, including a washed-out metastable average.
   */
  import { cellColor } from '../../lib/llm/colors';

  let {
    pattern,
    rows,
    cols,
    label = '',
    highlight = false,
    cellPx = 12,
    onCellClick
  }: {
    pattern: Float64Array;
    rows: number;
    cols: number;
    label?: string;
    /** Border in --chosen — "this is the pattern the weights favor". */
    highlight?: boolean;
    cellPx?: number;
    /** When supplied, cells become flippable (query editing). */
    onCellClick?: (index: number, value: number) => void;
  } = $props();

  const idx = $derived([...Array(rows * cols).keys()]);
</script>

<div class="pgrid" class:highlight>
  <div
    class="cells"
    style="grid-template-columns:repeat({cols}, {cellPx}px); grid-auto-rows:{cellPx}px"
  >
    {#each idx as i (i)}
      {@const v = pattern[i]}
      {#if onCellClick}
        <button
          class="cell clickable"
          style="background:{cellColor(v, 1, true)}"
          title={`[${Math.floor(i / cols)}, ${i % cols}] = ${v.toFixed(2)} — click to flip`}
          onclick={() => onCellClick(i, v)}
          aria-label={`flip pixel ${i}`}
        ></button>
      {:else}
        <div class="cell" style="background:{cellColor(v, 1, true)}" title={`[${Math.floor(i / cols)}, ${i % cols}] = ${v.toFixed(2)}`}></div>
      {/if}
    {/each}
  </div>
  {#if label}<div class="plabel mono faint">{label}</div>{/if}
</div>

<style>
  .pgrid {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    padding: 4px;
    border: 1.5px solid transparent;
    border-radius: 6px;
  }
  .pgrid.highlight { border-color: var(--chosen, #ffd166); }
  .cells {
    display: grid;
    gap: 1px;
    background: var(--bg-2);
    padding: 2px;
    border-radius: 4px;
  }
  .cell { border-radius: 1px; border: none; padding: 0; }
  .cell.clickable { cursor: pointer; }
  .cell.clickable:hover { outline: 1.5px solid var(--text); outline-offset: -1.5px; }
  .plabel { font-size: 10px; }
</style>
