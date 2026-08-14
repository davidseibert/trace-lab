<script lang="ts">
  import type { ScoredMove } from '../../lib/mdl/types';
  import {
    type MorphModel,
    type MergeMove,
    token,
    expandVisible,
    ruleIndexOf,
    isTerminal
  } from '../../lib/morphology/morphology';
  import { symColor } from '../../lib/morphology/colors';

  let { model, chosen }: { model: MorphModel; chosen: ScoredMove<MergeMove> | null } =
    $props();

  // For each word, which sequence positions are part of the next chosen merge,
  // so the user can SEE every word the merge folds at once.
  const highlights = $derived.by(() => {
    const sets = model.words.map(() => new Set<number>());
    if (!chosen) return sets;
    const { a, b } = chosen.move;
    model.words.forEach((w, wi) => {
      const set = sets[wi];
      const seq = w.seq;
      let i = 0;
      while (i < seq.length) {
        if (i + 1 < seq.length && seq[i] === a && seq[i + 1] === b) {
          set.add(i);
          set.add(i + 1);
          i += 2;
        } else i += 1;
      }
    });
    return sets;
  });

  const merges = $derived(
    highlights.reduce((s, set, wi) => s + (set.size / 2) * model.words[wi].count, 0)
  );
</script>

<div class="wordlist-view structure-view">
  <section class="block block--lex">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--model)"></span> Lexicon &nbsp;·&nbsp; L(M)</span>
      <span class="muted mono">{model.rules.length} morph{model.rules.length === 1 ? '' : 's'}</span>
    </div>
    {#if model.rules.length === 0}
      <p class="empty muted">No morphs yet — the lexicon is empty. Every word is spelled out as raw characters.</p>
    {:else}
      <div class="lex scrollbar">
        {#each model.rules as rule, k}
          {@const id = model.terminals.length + k}
          {@const c = symColor(model, id)}
          <div class="morph">
            <span class="chip morph-name" style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">M{k}</span>
            <span class="arrow">→</span>
            {#each rule.rhs as s}
              {@const cs = symColor(model, s)}
              <span class="chip" class:rulechip={!isTerminal(model, s)}
                    style="background:{cs.bg}; color:{cs.fg}; border-color:{cs.border}">{token(model, s)}</span>
            {/each}
            <span class="expand mono faint">“{expandVisible(model, id)}”</span>
          </div>
        {/each}
      </div>
    {/if}
  </section>

  <section class="block block--words">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--data)"></span> Segmentation &nbsp;·&nbsp; L(D|M)</span>
      <span class="muted mono">{model.words.length} types</span>
    </div>
    <div class="words scrollbar">
      {#each model.words as w, wi}
        <div class="word">
          <span class="freq mono faint" title="corpus frequency">×{w.count}</span>
          <div class="morphs">
            {#each w.seq as s, i}
              {@const c = symColor(model, s)}
              <span
                class="chip stream-chip"
                class:rulechip={!isTerminal(model, s)}
                class:hot={highlights[wi].has(i)}
                title={isTerminal(model, s) ? 'character' : `M${ruleIndexOf(model, s)} = “${expandVisible(model, s)}”`}
                style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">{token(model, s)}</span>
            {/each}
          </div>
        </div>
      {/each}
    </div>
    {#if chosen}
      <p class="next-note muted">
        Next: fold the
        <span class="mono" style="color:var(--chosen)">{chosen.label}</span>
        pair — {merges} weighted occurrence{merges === 1 ? '' : 's'} across the word list.
      </p>
    {/if}
  </section>
</div>

<style>
  /* Two full-height columns side by side: lexicon (model) | segmentation (data),
     each scrolling independently so later steps still show every entry.
     .block/.block-head/.words/.freq/.morphs + the chip vocabulary come from
     the global structure-view rules in app.css. */
  .wordlist-view { display: flex; flex-direction: row; gap: 12px; height: 100%; min-height: 0; }
  .block--lex { flex: 0.9 1 0; }
  .block--words { flex: 1.1 1 0; }
  .empty { font-size: 12.5px; margin: 4px 0; }

  .lex { display: flex; flex-direction: column; gap: 5px; overflow-y: auto; min-height: 0; flex: 1 1 auto; padding-right: 4px; }
  .morph { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .arrow { color: var(--faint); }
  .morph-name { font-weight: 700; }
  .expand { margin-left: 6px; font-size: 12px; }

  .word { display: flex; align-items: flex-start; gap: 8px; }

  @media (max-width: 720px) {
    .wordlist-view { flex-direction: column; }
    .block--lex { flex: 0 1 auto; max-height: 42%; }
  }

  .next-note { font-size: 12.5px; margin: 8px 0 0; }
</style>
