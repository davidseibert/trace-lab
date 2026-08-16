<script lang="ts" module>
  import type { Board } from '../../lib/tictac/game';

  export interface EquivItem {
    g: number;
    label: string;
    board: Board;
    pi: Float64Array;
    tv: number;
  }
</script>

<script lang="ts">
  /**
   * The current position under all 8 board symmetries, each with the model's
   * OWN policy for that transformed board — if the model were perfectly
   * D₄-equivariant, all eight overlays would be the same picture rotated.
   * The bar under each is the TV distance after pulling the policy back
   * through g⁻¹; the header meter is their mean.
   */
  import TicBoard from './TicBoard.svelte';

  let { items }: { items: EquivItem[] } = $props();
</script>

<div class="equiv scrollbar">
  {#each items as it (it.g)}
    <div class="gcell">
      <div class="glabel mono faint">{it.label}</div>
      <TicBoard board={it.board} pi={it.pi} cellPx={26} />
      <div class="tvtrack" title={`TV(g⁻¹·π(g·s), π(s)) = ${it.tv.toFixed(3)}`}>
        <div class="tvfill" style="width:{Math.min(100, it.tv * 100)}%"></div>
      </div>
      <div class="tvval mono faint">{it.g === 0 ? 'identity' : it.tv.toFixed(2)}</div>
    </div>
  {/each}
</div>

<style>
  .equiv { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; overflow: auto; min-height: 0; }
  .gcell { display: flex; flex-direction: column; align-items: center; gap: 3px; }
  .glabel { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.05em; }
  .tvtrack { width: 84px; height: 5px; background: var(--bg-2); border-radius: 3px; overflow: hidden; }
  .tvfill { height: 100%; background: var(--bad); border-radius: 3px; }
  .tvval { font-size: 9.5px; }
</style>
