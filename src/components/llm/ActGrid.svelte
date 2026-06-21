<script lang="ts">
  import type { TensorSnap } from '../../lib/llm/tensor';
  import { cellColor, maxAbs } from '../../lib/llm/colors';

  let {
    matrix,
    rowLabels = [],
    rowColors = [],
    signed = true,
    colLabel = 'dims'
  }: {
    matrix: TensorSnap;
    rowLabels?: string[];
    rowColors?: string[];
    /** Diverging (signed) vs single-hue intensity. */
    signed?: boolean;
    colLabel?: string;
  } = $props();

  const rows = $derived(matrix.shape[0]);
  const cols = $derived(matrix.shape[1]);
  const mx = $derived(maxAbs(matrix.data));
  const rowIdx = $derived([...Array(rows).keys()]);
  const colIdx = $derived([...Array(cols).keys()]);
</script>

<div class="actgrid scrollbar">
  {#each rowIdx as r}
    <div class="arow">
      {#if rowLabels.length}
        <span class="rlabel chip" style="color:{rowColors[r] ?? 'var(--muted)'}; border-color:{rowColors[r] ?? 'var(--border-2)'}">{rowLabels[r]}</span>
      {/if}
      <div class="cells" style="grid-template-columns:repeat({cols}, minmax(0, 1fr))">
        {#each colIdx as c}
          {@const v = matrix.data[r * cols + c]}
          <div class="cell" style="background:{cellColor(v, mx, signed)}" title={`[${r}, ${c}] = ${v.toFixed(3)}`}></div>
        {/each}
      </div>
    </div>
  {/each}
  <div class="axis faint mono">{rows} rows × {cols} {colLabel} · peak |{mx.toFixed(2)}|</div>
</div>

<style>
  .actgrid { display: flex; flex-direction: column; gap: 3px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .arow { display: flex; align-items: stretch; gap: 6px; min-height: 16px; }
  .rlabel {
    flex: 0 0 auto;
    align-self: center;
    min-width: 48px;
    text-align: right;
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 6px;
    border: 1px solid;
    border-radius: 4px;
    background: var(--bg-2);
    white-space: pre;
  }
  .cells { display: grid; gap: 2px; flex: 1 1 auto; min-width: 0; }
  .cell { aspect-ratio: 1 / 1; border-radius: 2px; min-height: 10px; }
  .axis { margin-top: 2px; font-size: 10px; }
</style>
