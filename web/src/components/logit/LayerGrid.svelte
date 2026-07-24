<script lang="ts">
  import type { TopTok } from '../../lib/logit/api';

  // The x-logit-lens TUI grid, reborn: rows are layers (deepest at top),
  // columns are input positions, each cell is that layer's top-1 guess for the
  // NEXT token, shaded by confidence. The current player row is highlighted;
  // clicking a cell seeks the player and selects the cell for the readout.
  let {
    layers,
    tokens,
    grid,
    nPrompt,
    index,
    selPos,
    onPick
  }: {
    layers: string[];
    tokens: string[];
    grid: TopTok[][][];
    /** tokens[:nPrompt] are the prompt; later columns are server-side rollout. */
    nPrompt: number;
    /** Current player row (layer index, 0 = embed). */
    index: number;
    /** Selected position column. */
    selPos: number;
    onPick: (row: number, pos: number) => void;
  } = $props();

  // Deepest layer on top, like the TUI (and like reading a lens ladder).
  const rows = $derived(layers.map((name, r) => ({ name, r })).reverse());

  const cellBg = (p: number) => `rgba(91, 156, 255, ${(0.06 + p * 0.5).toFixed(3)})`;
</script>

<div class="wrap scrollbar">
  <table class="mono">
    <thead>
      <tr>
        <th class="lname"></th>
        {#each tokens as t, p (p)}
          <th
            class="pos"
            class:sel={p === selPos}
            class:gen={p >= nPrompt}
            title={p >= nPrompt ? `rollout position ${p} (model-generated)` : `input position ${p}`}
          >{t}</th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as { name, r } (r)}
        <tr class:cur={r === index}>
          <td class="lname" title="seek to {name}">{name}</td>
          {#each grid[r] as cell, p (p)}
            {@const top = cell[0]}
            <td
              class="cell"
              class:sel={r === index && p === selPos}
              style="background:{cellBg(top.p)}"
              title={`${name} @ “${tokens[p]}” → ${top.t} (${(top.p * 100).toFixed(1)}%)`}
              onclick={() => onPick(r, p)}
            >
              {top.t}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
  <p class="hint faint">
    each cell: that layer's guess for the token <i>after</i> that column — click to inspect
  </p>
</div>

<style>
  .wrap { overflow: auto; min-height: 0; flex: 1 1 auto; }
  table { border-collapse: collapse; font-size: 10.5px; }
  th, td { padding: 2px 7px; text-align: left; white-space: nowrap; }

  th.pos {
    color: var(--muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--panel);
  }
  th.pos.sel { color: var(--text); }
  th.pos.gen { color: var(--data); font-style: italic; }

  td.lname, th.lname {
    color: var(--faint);
    padding-right: 10px;
    position: sticky;
    left: 0;
    background: var(--panel);
  }
  tr.cur td.lname { color: var(--model); }

  td.cell { cursor: pointer; border: 1px solid transparent; border-radius: 3px; }
  td.cell:hover { border-color: var(--border-2); }
  tr.cur td.cell { border-top-color: rgba(91, 156, 255, 0.35); border-bottom-color: rgba(91, 156, 255, 0.35); }
  td.cell.sel { border-color: var(--model); }

  .hint { font-size: 10px; margin: 6px 2px 0; }
</style>
