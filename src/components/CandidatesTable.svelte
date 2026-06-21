<script lang="ts">
  import type { CostBreakdown, ScoredMove } from '../lib/mdl/types';
  import { fmt, fmtDelta } from '../lib/mdl/format';

  // Both lenses' moves are an adjacent symbol pair (a,b): grammar's DigramMove
  // and morphology's MergeMove. We only need that structural shape here.
  type PairMove = { a: number; b: number };

  let {
    candidates,
    baseline,
    chosen,
    limit = 40
  }: {
    candidates: ScoredMove<PairMove>[];
    baseline: CostBreakdown;
    chosen: ScoredMove<PairMove> | null;
    limit?: number;
  } = $props();

  const sameMove = (m: PairMove | undefined, n: PairMove | undefined) =>
    !!m && !!n && m.a === n.a && m.b === n.b;

  // The candidate a frequency-greedy compressor (e.g. plain RePair) would pick.
  const mostFrequent = $derived.by(() => {
    let best: ScoredMove<PairMove> | null = null;
    let bestN = -1;
    for (const c of candidates) {
      const n = Number(c.extra['×'] ?? 0);
      if (n > bestN) { bestN = n; best = c; }
    }
    return best;
  });

  const shown = $derived(candidates.slice(0, limit));
</script>

<div class="cands">
  {#if candidates.length === 0}
    <p class="muted empty">No digram repeats ≥ 2×. Nothing left to compress — converged.</p>
  {:else}
    <div class="table scrollbar">
      <div class="row head">
        <span>move</span>
        <span class="r">×</span>
        <span>expands</span>
        <span class="r">ΔL(M)</span>
        <span class="r">ΔL(D|M)</span>
        <span class="r">Δ total</span>
      </div>
      {#each shown as c}
        {@const isChosen = sameMove(c.move, chosen?.move)}
        {@const isFreq = sameMove(c.move, mostFrequent?.move)}
        <div class="row" class:chosen={isChosen}>
          <span class="mv mono">
            {#if isChosen}<span class="tag pick">PICK</span>{/if}
            {c.label}
          </span>
          <span class="r mono">
            {c.extra['×']}
            {#if isFreq && !isChosen}<span class="tag freq" title="what a frequency-greedy compressor would pick">freq</span>{/if}
          </span>
          <span class="exp mono faint">{c.extra['expands']}</span>
          <span class="r mono" class:bad={c.modelBitsAfter - baseline.modelBits > 0}>
            {fmtDelta(c.modelBitsAfter - baseline.modelBits)}
          </span>
          <span class="r mono" class:good={c.dataBitsAfter - baseline.dataBits < 0}>
            {fmtDelta(c.dataBitsAfter - baseline.dataBits)}
          </span>
          <span class="r mono total" class:good={c.delta < 0} class:bad={c.delta > 0}>
            {fmtDelta(c.delta)}
          </span>
        </div>
      {/each}
      {#if candidates.length > limit}
        <div class="more faint">… {candidates.length - limit} more not shown</div>
      {/if}
    </div>
    {#if mostFrequent && chosen && !sameMove(mostFrequent.move, chosen.move)}
      <p class="insight">
        💡 MDL picked <span class="mono" style="color:var(--chosen)">{chosen.label}</span>
        (Δ {fmt(chosen.delta, 1)}), <em>not</em> the most frequent digram
        <span class="mono">{mostFrequent.label}</span> (Δ {fmt(mostFrequent.delta, 1)}).
        Frequency ≠ best compression.
      </p>
    {/if}
  {/if}
</div>

<style>
  .cands { display: flex; flex-direction: column; gap: 7px; min-height: 0; height: 100%; }
  .empty { font-size: 12.5px; }
  .table { display: flex; flex-direction: column; overflow-y: auto; min-height: 0; flex: 1 1 auto; font-size: 12px; }
  .row {
    display: grid;
    grid-template-columns: 1.6fr 0.7fr 1.3fr 0.9fr 1fr 0.9fr;
    gap: 8px; align-items: center;
    padding: 3px 6px; border-radius: 4px;
  }
  .row.head { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; position: sticky; top: 0; background: var(--panel); }
  .row:not(.head):hover { background: var(--panel-2); }
  .row.chosen { background: rgba(255, 209, 102, 0.1); outline: 1px solid rgba(255, 209, 102, 0.3); }
  .r { text-align: right; justify-self: end; }
  .total { font-weight: 700; }
  .exp { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tag { font-size: 9px; font-weight: 700; padding: 1px 4px; border-radius: 3px; letter-spacing: 0.04em; margin-right: 4px; }
  .tag.pick { background: var(--chosen); color: #1a1400; }
  .tag.freq { background: var(--border-2); color: var(--muted); margin-left: 4px; }
  .more { padding: 6px; font-size: 11px; }
  .insight { flex: 0 0 auto; font-size: 11.5px; line-height: 1.4; background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 7px 9px; margin: 0; }
</style>
