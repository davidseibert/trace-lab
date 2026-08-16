<script lang="ts">
  /**
   * Round-robin standings: rows play X, columns play O, each cell that
   * match's w-d-l from X's point of view, tinted by who came out ahead.
   * Click a cell to load its games into the match panel for replay.
   */
  import type { MatchResult } from '../../lib/tictac/arena';

  let {
    labels,
    results,
    onPick
  }: {
    /** Player id → display label, in roster order. */
    labels: [string, string][];
    results: MatchResult[];
    onPick?: (r: MatchResult) => void;
  } = $props();

  const byPair = $derived.by(() => {
    const m = new Map<string, MatchResult>();
    for (const r of results) m.set(`${r.x}|${r.o}`, r);
    return m;
  });

  const tint = (r: MatchResult) =>
    r.xWins > r.oWins ? 'rgba(87,217,163,0.18)' : r.xWins < r.oWins ? 'rgba(255,107,107,0.18)' : 'rgba(139,147,167,0.10)';
</script>

<div class="wrap scrollbar">
  <table class="mono">
    <thead>
      <tr>
        <th class="corner faint">X ↓ · O →</th>
        {#each labels as [id, label] (id)}<th title={id}>{label}</th>{/each}
      </tr>
    </thead>
    <tbody>
      {#each labels as [xid, xlabel] (xid)}
        <tr>
          <th class="rowh" title={xid}>{xlabel}</th>
          {#each labels as [oid] (oid)}
            {#if xid === oid}
              <td class="self">—</td>
            {:else}
              {@const r = byPair.get(`${xid}|${oid}`)}
              {#if r}
                <td
                  class="cell"
                  class:pickable={!!onPick}
                  style="background:{tint(r)}"
                  title={`${xlabel} (X) vs ${oid} (O): ${r.xWins}-${r.draws}-${r.oWins} — click to load games`}
                  onclick={() => onPick?.(r)}
                >{r.xWins}-{r.draws}-{r.oWins}</td>
              {:else}
                <td class="faint">·</td>
              {/if}
            {/if}
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
  <div class="faint hint">cells are X's w-draw-l · green = row player ahead, red = column player ahead</div>
</div>

<style>
  .wrap { overflow: auto; min-height: 0; display: flex; flex-direction: column; gap: 6px; }
  table { border-collapse: collapse; font-size: 11px; }
  th, td { padding: 4px 8px; border: 1px solid var(--border-2); text-align: center; white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; }
  .corner { font-size: 9.5px; }
  .rowh { text-align: right; }
  .self { color: var(--border-2); }
  .cell.pickable { cursor: pointer; }
  .cell.pickable:hover { outline: 1.5px solid var(--text); outline-offset: -1.5px; }
  .hint { font-size: 10.5px; }
</style>
