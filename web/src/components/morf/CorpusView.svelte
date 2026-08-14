<script lang="ts">
  import { type MorfState, lexiconRows, showMorph } from '../../lib/morfessor/morfessor';
  import { morphColor } from '../../lib/morfessor/colors';

  let { model, focusWord }: { model: MorfState; focusWord: number | null } = $props();

  const rows = $derived(lexiconRows(model));
  const maxCount = $derived(Math.max(1, ...rows.map((r) => r.count)));
</script>

<div class="corpus-view structure-view">
  <section class="block block--lex">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--model)"></span> Lexicon &nbsp;·&nbsp; L(M)</span>
      <span class="muted mono">{rows.length} morph{rows.length === 1 ? '' : 's'}</span>
    </div>
    <div class="lex scrollbar">
      {#each rows as r}
        {@const c = morphColor(r.morph)}
        <div class="lexrow">
          <span class="chip" class:rulechip={r.morph.length > 1}
                style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">{showMorph(r.morph)}</span>
          <span class="bar" title="token count">
            <span class="fill" style="width:{(r.count / maxCount) * 100}%"></span>
          </span>
          <span class="cnt mono faint" title="weighted token count">{r.count}</span>
          <span class="cost mono faint" title="lexicon spelling cost">{r.bits.toFixed(1)}b</span>
        </div>
      {/each}
    </div>
  </section>

  <section class="block block--words">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--data)"></span> Corpus &nbsp;·&nbsp; L(D|M)</span>
      <span class="muted mono">{model.words.length} types</span>
    </div>
    <div class="words scrollbar">
      {#each model.words as w, wi}
        <div class="word" class:focus={wi === focusWord}>
          <span class="freq mono faint" title="corpus frequency">×{w.count}</span>
          <div class="morphs">
            {#each model.analyses[wi] as m}
              {@const c = morphColor(m)}
              <span class="chip stream-chip" class:rulechip={m.length > 1}
                    style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">{showMorph(m)}</span>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  </section>
</div>

<style>
  /* Lexicon (model) | corpus (data), side by side, each scrolling on its own.
     .block/.block-head/.words/.freq/.morphs + the chip vocabulary come from
     the global structure-view rules in app.css. */
  .corpus-view { display: flex; flex-direction: row; gap: 12px; height: 100%; min-height: 0; }
  .block--lex { flex: 1 1 0; }
  .block--words { flex: 1 1 0; }

  .lex { display: flex; flex-direction: column; gap: 3px; overflow-y: auto; min-height: 0; flex: 1 1 auto; padding-right: 4px; }
  .lexrow { display: grid; grid-template-columns: auto 1fr auto auto; gap: 7px; align-items: center; padding: 1px 0; }
  .bar { height: 6px; background: var(--bg-2); border-radius: 3px; overflow: hidden; min-width: 24px; }
  .fill { display: block; height: 100%; background: var(--data); opacity: 0.7; }
  .cnt { font-size: 11px; min-width: 22px; text-align: right; }
  .cost { font-size: 10.5px; min-width: 34px; text-align: right; }

  .word { display: flex; align-items: flex-start; gap: 8px; padding: 2px 4px; border-radius: 4px; }
  .word.focus { background: rgba(255, 209, 102, 0.12); outline: 1px solid rgba(255, 209, 102, 0.35); }

  @media (max-width: 720px) {
    .corpus-view { flex-direction: column; }
    .block--lex { flex: 0 1 auto; max-height: 42%; }
  }
</style>
