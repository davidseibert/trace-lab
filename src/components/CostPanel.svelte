<script lang="ts">
  import type { CostBreakdown } from '../lib/mdl/types';
  import { fmt, fmtBits } from '../lib/mdl/format';

  let { cost, reference }: { cost: CostBreakdown; reference: number } = $props();

  // Scale bars to the initial (largest) total so shrinkage is visible over time.
  const scale = $derived(reference > 0 ? 100 / reference : 0);
  const savedPct = $derived(reference > 0 ? (1 - cost.total / reference) * 100 : 0);
</script>

<div class="cost">
  <div class="headline">
    <div class="total">
      <span class="num mono" style="color:var(--total)">{fmt(cost.total, 1)}</span>
      <span class="unit muted">bits total</span>
    </div>
    <div class="saved mono" class:good={savedPct > 0}>
      {savedPct >= 0 ? '−' : '+'}{fmt(Math.abs(savedPct), 1)}% vs. raw
    </div>
  </div>

  <div class="bar" style="width:{Math.max(cost.total * scale, 0.5)}%">
    <div class="seg model" style="flex:{cost.modelBits}" title="L(M) = {fmtBits(cost.modelBits)}"></div>
    <div class="seg data" style="flex:{cost.dataBits}" title="L(D|M) = {fmtBits(cost.dataBits)}"></div>
  </div>

  <div class="legend">
    <div class="leg">
      <span class="swatch" style="background:var(--model)"></span>
      <span class="muted">L(M) model</span>
      <span class="mono">{fmt(cost.modelBits, 1)}</span>
    </div>
    <div class="leg">
      <span class="swatch" style="background:var(--data)"></span>
      <span class="muted">L(D|M) data</span>
      <span class="mono">{fmt(cost.dataBits, 1)}</span>
    </div>
  </div>

  <div class="terms">
    {#each [...cost.modelTerms.map((t) => ({ t, kind: 'model' })), ...cost.dataTerms.map((t) => ({ t, kind: 'data' }))] as { t, kind }}
      <div class="term" class:fixed={t.fixed}>
        <span class="dot" style="background:var(--{kind})"></span>
        <span class="tlabel">{t.label}{#if t.fixed}<span class="faint"> (fixed)</span>{/if}</span>
        <span class="tbits mono">{fmt(t.bits, 1)}</span>
        {#if t.detail}<span class="tdetail faint mono">{t.detail}</span>{/if}
      </div>
    {/each}
  </div>

  <div class="meta">
    {#each Object.entries(cost.meta) as [k, v]}
      <div class="mcell">
        <span class="mlabel muted">{k}</span>
        <span class="mval mono">{Number.isInteger(v) ? v : fmt(v)}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .cost { display: flex; flex-direction: column; gap: 9px; }
  .headline { display: flex; align-items: baseline; justify-content: space-between; }
  .total { display: flex; align-items: baseline; gap: 6px; }
  .num { font-size: 22px; font-weight: 700; }
  .unit { font-size: 11px; }
  .saved { font-size: 12px; color: var(--muted); }

  .bar {
    display: flex; height: 18px; border-radius: 5px; overflow: hidden;
    background: var(--bg-2); border: 1px solid var(--border);
    transition: width 0.3s ease;
    min-width: 4px;
  }
  .seg { transition: flex-grow 0.3s ease; }
  .seg.model { background: var(--model); }
  .seg.data { background: var(--data); }

  .legend { display: flex; gap: 18px; }
  .leg { display: flex; align-items: center; gap: 6px; font-size: 12.5px; }
  .leg .mono { margin-left: 2px; }

  .terms { display: flex; flex-direction: column; gap: 3px; border-top: 1px solid var(--border); padding-top: 8px; }
  .term { display: grid; grid-template-columns: 9px 1fr auto; gap: 7px; align-items: baseline; font-size: 12px; }
  .term.fixed { opacity: 0.6; }
  .dot { width: 7px; height: 7px; border-radius: 2px; align-self: center; }
  .tlabel { color: var(--text); }
  .tbits { color: var(--text); }
  .tdetail { grid-column: 2 / 4; font-size: 10.5px; margin-top: -2px; line-height: 1.3; }

  .meta { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 14px; border-top: 1px solid var(--border); padding-top: 8px; }
  .mcell { display: flex; justify-content: space-between; font-size: 12px; }
  .mval { color: var(--total); }
</style>
