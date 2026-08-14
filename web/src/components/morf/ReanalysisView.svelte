<script lang="ts">
  import type { MorfStep } from '../../lib/morfessor/morfessor';
  import { showMorph } from '../../lib/morfessor/morfessor';
  import { morphColor } from '../../lib/morfessor/colors';
  import { fmtDelta } from '../../lib/mdl/format';

  let { step }: { step: MorfStep } = $props();

  const sameSeg = (a: string[], b: string[]) =>
    a.length === b.length && a.every((m, i) => m === b[i]);

  const focus = $derived(step.focusWord !== null ? step.model.words[step.focusWord] : null);
</script>

<div class="reanalysis structure-view">
  {#if !focus}
    <p class="muted empty">
      Converged — a full pass over the word list found no cut that lowers the total.
      This segmentation is the MDL local optimum.
    </p>
  {:else}
    <div class="head">
      <span class="muted">re-analysing</span>
      <span class="surface mono">{showMorph(focus.surface)}</span>
      <span class="muted mono faint">×{focus.count}</span>
      <span class="muted">— every segmentation, scored against the rest of the model:</span>
    </div>

    <div class="table scrollbar">
      <div class="row head-row">
        <span>segmentation</span>
        <span class="r">morphs</span>
        <span class="r">Δ total</span>
      </div>
      {#each step.candidates as c}
        {@const isChosen = !!step.chosen && sameSeg(c.seg, step.chosen.seg)}
        {@const isCurrent = sameSeg(c.seg, step.oldSeg)}
        <div class="row" class:chosen={isChosen}>
          <span class="seg">
            {#if isChosen}<span class="tag pick">PICK</span>{/if}
            {#each c.seg as m}
              {@const col = morphColor(m)}
              <span class="chip" class:rulechip={m.length > 1}
                    style="background:{col.bg}; color:{col.fg}; border-color:{col.border}">{showMorph(m)}</span>
            {/each}
            {#if isCurrent && !isChosen}<span class="tag cur" title="the current segmentation">current</span>{/if}
          </span>
          <span class="r mono faint">{c.seg.length}</span>
          <span class="r mono total" class:good={c.delta < -1e-9} class:bad={c.delta > 1e-9}>
            {Math.abs(c.delta) < 1e-9 ? '0.00' : fmtDelta(c.delta)}
          </span>
        </div>
      {/each}
    </div>

    <p class="insight">
      {#if step.changed}
        💡 A later epoch can still re-cut this word: its best segmentation depends on
        every other word's morph counts, which keep shifting until the sweep settles.
      {:else}
        Keeping the current cut — no alternative lowers the total <em>given the rest of
        the model as it stands now</em>.
      {/if}
    </p>
  {/if}
</div>

<style>
  .reanalysis { display: flex; flex-direction: column; gap: 8px; min-height: 0; height: 100%; }
  .empty { font-size: 12.5px; line-height: 1.5; }
  .head { display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; font-size: 12px; flex: 0 0 auto; }
  .surface { font-size: 14px; font-weight: 700; color: var(--text); }

  .table { display: flex; flex-direction: column; overflow-y: auto; min-height: 0; flex: 1 1 auto; font-size: 12px; }
  /* .row.chosen, .tag/.tag.pick and the chip vocabulary come from the global
     structure-view rules in app.css. */
  .row { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; padding: 3px 6px; border-radius: 4px; }
  .head-row { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; position: sticky; top: 0; background: var(--panel); }
  .row:not(.head-row):hover { background: var(--panel-2); }
  .r { text-align: right; justify-self: end; }
  .total { font-weight: 700; }
  .seg { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; min-width: 0; }

  .tag.cur { background: var(--border-2); color: var(--muted); margin-left: 4px; }

  .insight { flex: 0 0 auto; font-size: 11.5px; line-height: 1.4; background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 7px 9px; margin: 0; }
</style>
