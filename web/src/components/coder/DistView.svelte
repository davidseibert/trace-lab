<script lang="ts">
  import type { CoderStep } from '../../lib/coder/arithmetic';
  import { symColorFor } from '../../lib/coder/colors';
  import { surprisal } from '../../lib/mdl/format';

  let { step }: { step: CoderStep } = $props();

  // Distribution rows with their cumulative ranges (what the coder splits on).
  const rows = $derived.by(() => {
    let cum = 0;
    return step.dist.map((e, i) => {
      const lo = cum;
      cum += e.p;
      return {
        e,
        i,
        lo,
        hi: cum,
        bits: surprisal(e.p),
        chosen: e.id === step.dist[step.chosenIndex].id
      };
    });
  });
</script>

<div class="dist">
  <div class="stack">
    {#each rows as r (r.e.id)}
      {@const c = symColorFor(r.e.id)}
      <div
        class="seg"
        class:chosen={r.chosen}
        style="flex:{r.e.p}; background:{c.bg}; color:{c.fg}; border-color:{c.border}"
        title="{r.e.label}  p={r.e.p.toFixed(3)}  [{r.lo.toFixed(3)}, {r.hi.toFixed(3)})"
      >
        {#if r.e.p > 0.06}<span class="slab mono">{r.e.label}</span>{/if}
      </div>
    {/each}
  </div>

  <div class="rows scrollbar">
    {#each rows as r (r.e.id)}
      {@const c = symColorFor(r.e.id)}
      <div class="row" class:chosen={r.chosen}>
        <span class="sw" style="background:{c.bg}; border-color:{c.border}"></span>
        <span class="tok mono" style="color:{c.fg}">{r.e.label}</span>
        <span class="p mono">{(r.e.p * 100).toFixed(1)}%</span>
        <span class="range mono faint">[{r.lo.toFixed(3)}, {r.hi.toFixed(3)})</span>
        <span class="bits mono">{r.bits.toFixed(2)} b</span>
        {#if r.chosen}<span class="tag">EMIT</span>{/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .dist { display: flex; flex-direction: column; gap: 8px; flex: 1 1 auto; min-height: 0; }
  .stack { display: flex; height: 22px; border-radius: var(--r-sm); overflow: hidden; border: 1px solid var(--border); flex: 0 0 auto; }
  .seg { display: flex; align-items: center; justify-content: center; min-width: 1px; border-right: 1px solid rgba(0,0,0,0.25); opacity: 0.45; transition: opacity 0.2s ease; }
  .seg.chosen { opacity: 1; outline: 2px solid; outline-offset: -2px; }
  .slab { font-size: 10px; }

  .rows { display: flex; flex-direction: column; gap: 2px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .row { display: grid; grid-template-columns: 14px 40px 52px 1fr 56px 44px; gap: 8px; align-items: center; padding: 2px 4px; border-radius: 4px; font-size: 12px; }
  .row.chosen { background: rgba(255, 209, 102, 0.1); outline: 1px solid rgba(255, 209, 102, 0.3); }
  .sw { width: 11px; height: 11px; border-radius: 2px; border: 1px solid; }
  .tok { text-align: right; }
  .p { text-align: right; color: var(--muted); }
  .range { font-size: 11px; }
  .bits { text-align: right; color: var(--data); }
  .tag { font-size: 9px; font-weight: 700; padding: 1px 4px; border-radius: 3px; background: var(--chosen); color: #1a1400; letter-spacing: 0.04em; }
</style>
