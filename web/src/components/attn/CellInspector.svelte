<script lang="ts">
  import type { AttnTrace } from '../../lib/attn/attention';

  let {
    trace,
    tokens,
    selected
  }: {
    trace: AttnTrace;
    tokens: string[];
    selected: { head: number; row: number; col: number } | null;
  } = $props();

  const head = $derived(selected ? trace.heads[selected.head] : null);
  const headDim = $derived(trace.headDim);

  const qVec = $derived.by(() => {
    if (!head || !selected) return [];
    const off = selected.row * headDim;
    return Array.from(head.Q.data.subarray(off, off + headDim));
  });
  const kVec = $derived.by(() => {
    if (!head || !selected) return [];
    const off = selected.col * headDim;
    return Array.from(head.K.data.subarray(off, off + headDim));
  });
  const products = $derived(qVec.map((q, i) => q * kVec[i]));
  const score = $derived(selected && head ? head.scores.data[selected.row * trace.T + selected.col] : 0);
  const scaled = $derived(selected && head ? head.scaled.data[selected.row * trace.T + selected.col] : 0);
  const weight = $derived(selected && head ? head.weights.data[selected.row * trace.T + selected.col] : 0);
  const masked = $derived(!!(selected && trace.cfg.causal && selected.col > selected.row));
  const sqrtHeadDim = $derived(Math.sqrt(headDim));
</script>

<div class="inspector">
  {#if !selected || !head}
    <p class="say muted">
      Click any cell in <b>Scores</b> or <b>Attention weights</b> to see the exact dot product
      behind that number.
    </p>
  {:else}
    <p class="say">
      head <b>{selected.head}</b> · <span class="mono" style="color:var(--model)">{tokens[selected.row]}</span>
      (row {selected.row}) attending to
      <span class="mono" style="color:var(--data)">{tokens[selected.col]}</span> (col {selected.col})
    </p>

    <div class="dot mono">
      {#each qVec as q, i}<span class="term">{i > 0 ? ' + ' : ''}({q.toFixed(2)})({kVec[i].toFixed(2)})</span>{/each}
      <span class="eq"> = {score.toFixed(3)}</span>
    </div>
    <p class="say muted">Q{selected.row} · K{selected.col} — one dot product per (query, key) pair, over the {headDim} dims of this head.</p>

    <div class="dot mono">{score.toFixed(3)} ÷ √{headDim} = {scaled.toFixed(3)}</div>
    <p class="say muted">Scaling keeps dot products from blowing up softmax as head dim grows.</p>

    {#if masked}
      <p class="say bad">
        Masked — causal attention forbids <span class="mono">{tokens[selected.row]}</span> (position {selected.row})
        from looking at <span class="mono">{tokens[selected.col]}</span> (position {selected.col}), a later token. This
        score never enters the softmax; the weight is 0.
      </p>
    {:else}
      <div class="dot mono">softmax(row {selected.row})[{selected.col}] = {weight.toFixed(3)} <span class="pct">({(weight * 100).toFixed(0)}%)</span></div>
      <p class="say muted">
        The whole row of scores is exponentiated and normalised together, so this weight depends on every
        other key in {trace.cfg.causal ? 'this token’s causal window' : 'the sequence'}, not just this one pair.
      </p>
    {/if}
  {/if}
</div>

<style>
  .inspector { display: flex; flex-direction: column; gap: 6px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .say { margin: 0; font-size: 12.5px; line-height: 1.45; }
  .say.muted { color: var(--muted); }
  .say.bad { color: var(--bad); }
  .dot { font-size: 12px; padding: 6px 8px; background: var(--bg-2); border-radius: var(--r-sm); overflow-x: auto; white-space: nowrap; }
  .term { color: var(--muted); }
  .eq { color: var(--text); font-weight: 600; }
  .pct { color: var(--muted); }
</style>
