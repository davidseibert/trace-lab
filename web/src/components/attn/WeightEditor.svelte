<script lang="ts">
  import type { TensorSnap } from '../../lib/llm/tensor';
  import { cellColor, maxAbs } from '../../lib/llm/colors';

  let {
    value,
    onChange,
    rowLabels = []
  }: {
    value: TensorSnap;
    onChange: (next: TensorSnap) => void;
    rowLabels?: string[];
  } = $props();

  const rows = $derived(value.shape[0]);
  const cols = $derived(value.shape[1]);
  const mx = $derived(maxAbs(value.data));
  const rowIdx = $derived([...Array(rows).keys()]);
  const colIdx = $derived([...Array(cols).keys()]);

  function setCell(r: number, c: number, raw: string) {
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    const data = new Float64Array(value.data);
    data[r * cols + c] = n;
    onChange({ data, shape: [...value.shape] });
  }
</script>

<div class="weditor scrollbar">
  {#each rowIdx as r}
    <div class="wrow">
      {#if rowLabels.length}<span class="rlabel mono">{rowLabels[r]}</span>{/if}
      <div class="wcells" style="grid-template-columns:repeat({cols}, minmax(0, 1fr))">
        {#each colIdx as c}
          {@const v = value.data[r * cols + c]}
          <input
            class="wcell mono"
            type="number"
            step="0.1"
            style="background:{cellColor(v, mx, true)}"
            value={v.toFixed(2)}
            onchange={(e) => setCell(r, c, (e.currentTarget as HTMLInputElement).value)}
          />
        {/each}
      </div>
    </div>
  {/each}
</div>

<style>
  .weditor { display: flex; flex-direction: column; gap: 3px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .wrow { display: flex; align-items: stretch; gap: 6px; min-height: 20px; }
  .rlabel {
    flex: 0 0 auto;
    align-self: center;
    min-width: 48px;
    text-align: right;
    font-size: 11px;
    padding: 2px 6px;
    color: var(--muted);
    white-space: pre;
  }
  .wcells { display: grid; gap: 2px; flex: 1 1 auto; min-width: 0; }
  .wcell {
    width: 100%;
    min-width: 0;
    aspect-ratio: 2 / 1;
    border: 1px solid var(--border-2);
    border-radius: 2px;
    padding: 0 2px;
    text-align: center;
    font-size: 10px;
    color: var(--text);
    appearance: textfield;
    -moz-appearance: textfield;
  }
  .wcell::-webkit-outer-spin-button,
  .wcell::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .wcell:focus { outline: 1.5px solid var(--model); outline-offset: -1.5px; }
</style>
