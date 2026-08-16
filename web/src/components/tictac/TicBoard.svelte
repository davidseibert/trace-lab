<script lang="ts">
  /**
   * The 3×3 board: marks for occupied cells, and on empty cells the model's
   * (legal-renormalized) probability as a blue fill — so the policy is read
   * directly off the board. Optimal moves ring green (solver truth), the
   * model's own pick outlines yellow; agreement is visible as rings and
   * outline landing on the same cell.
   */
  import type { Board } from '../../lib/tictac/game';

  let {
    board,
    pi = null,
    optimal = [],
    modelPick = null,
    cellPx = 56,
    onCellClick
  }: {
    board: Board;
    /** Legal-renormalized policy over cells; null hides the overlay. */
    pi?: Float64Array | null;
    optimal?: number[];
    modelPick?: number | null;
    cellPx?: number;
    onCellClick?: (cell: number) => void;
  } = $props();

  const maxPi = $derived(pi ? Math.max(1e-9, ...pi) : 1);
</script>

<div class="tboard" style="grid-template-columns: repeat(3, {cellPx}px); grid-auto-rows: {cellPx}px">
  {#each board as c, i (i)}
    {#if c !== 0}
      <div class="cell mark" class:x={c === 1} class:o={c === 2} style="font-size:{cellPx * 0.55}px">
        {c === 1 ? '✕' : '◯'}
      </div>
    {:else}
      <button
        class="cell empty"
        class:optimal={optimal.includes(i)}
        class:pick={modelPick === i}
        class:clickable={!!onCellClick}
        style="background: rgba(91,156,255,{pi ? ((0.85 * pi[i]) / maxPi).toFixed(3) : 0})"
        title={pi ? `cell ${i} · π = ${(pi[i] * 100).toFixed(1)}%${optimal.includes(i) ? ' · optimal' : ''}` : `cell ${i}`}
        onclick={() => onCellClick?.(i)}
        disabled={!onCellClick}
        aria-label={`play cell ${i}`}
      >
        {#if pi && cellPx >= 40}<span class="pct mono">{(pi[i] * 100).toFixed(0)}</span>{/if}
      </button>
    {/if}
  {/each}
</div>

<style>
  .tboard { display: grid; gap: 3px; background: var(--border-2); padding: 3px; border-radius: 6px; width: fit-content; }
  .cell {
    display: flex; align-items: center; justify-content: center;
    background: var(--bg-2); border: none; border-radius: 3px; padding: 0;
    position: relative;
  }
  .mark.x { color: var(--model); }
  .mark.o { color: var(--data); }
  .empty.clickable { cursor: pointer; }
  .empty.clickable:hover { outline: 1.5px solid var(--text); outline-offset: -1.5px; }
  .empty.optimal { box-shadow: inset 0 0 0 2px var(--good); }
  .empty.pick { outline: 2px solid var(--chosen); outline-offset: -2px; }
  .pct { font-size: 10px; color: var(--muted); pointer-events: none; }
</style>
