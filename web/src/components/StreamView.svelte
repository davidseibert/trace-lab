<script lang="ts">
  import type { ScoredMove } from '../lib/mdl/types';
  import {
    type GrammarModel,
    type DigramMove,
    token,
    expandVisible,
    ruleIndexOf,
    isTerminal
  } from '../lib/string/grammar';
  import { symColor } from '../lib/string/colors';

  let { model, chosen }: { model: GrammarModel; chosen: ScoredMove<DigramMove> | null } =
    $props();

  // Mark which sequence positions are part of the next chosen digram, so the
  // user can SEE what is about to be folded into a rule.
  const highlight = $derived.by(() => {
    const set = new Set<number>();
    if (!chosen) return set;
    const { a, b } = chosen.move;
    const seq = model.sequence;
    let i = 0;
    while (i < seq.length) {
      if (i + 1 < seq.length && seq[i] === a && seq[i + 1] === b) {
        set.add(i);
        set.add(i + 1);
        i += 2;
      } else i += 1;
    }
    return set;
  });
</script>

<div class="stream-view">
  <section class="block block--rules">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--model)"></span> Dictionary &nbsp;·&nbsp; L(M)</span>
      <span class="muted mono">{model.rules.length} rule{model.rules.length === 1 ? '' : 's'}</span>
    </div>
    {#if model.rules.length === 0}
      <p class="empty muted">No rules yet — the model is empty. Every symbol in the data is a raw character.</p>
    {:else}
      <div class="rules scrollbar">
        {#each model.rules as rule, k}
          {@const id = model.terminals.length + k}
          {@const c = symColor(model, id)}
          <div class="rule">
            <span class="chip rule-name" style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">R{k}</span>
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

  <section class="block block--stream">
    <div class="block-head">
      <span class="panel-title"><span class="swatch" style="background:var(--data)"></span> Compressed stream &nbsp;·&nbsp; L(D|M)</span>
      <span class="muted mono">{model.sequence.length} symbols</span>
    </div>
    <div class="stream scrollbar">
      {#each model.sequence as s, i}
        {@const c = symColor(model, s)}
        <span
          class="chip stream-chip"
          class:rulechip={!isTerminal(model, s)}
          class:hot={highlight.has(i)}
          title={isTerminal(model, s) ? 'character' : `R${ruleIndexOf(model, s)} = “${expandVisible(model, s)}”`}
          style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">{token(model, s)}</span>
      {/each}
    </div>
    {#if chosen}
      <p class="next-note muted">
        Next: fold the {highlight.size / 2}× highlighted
        <span class="mono" style="color:var(--chosen)">{chosen.label}</span>
        into a new rule.
      </p>
    {/if}
  </section>
</div>

<style>
  .stream-view { display: flex; flex-direction: column; gap: 10px; height: 100%; min-height: 0; }
  .block { display: flex; flex-direction: column; min-height: 0; }
  .block--rules { flex: 0 1 auto; max-height: 42%; }
  .block--stream { flex: 1 1 auto; }
  .block-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex: 0 0 auto; }
  .empty { font-size: 12.5px; margin: 4px 0; }

  .rules { display: flex; flex-direction: column; gap: 5px; overflow-y: auto; min-height: 0; padding-right: 4px; }
  .rule { display: flex; align-items: center; gap: 6px; }
  .arrow { color: var(--faint); }
  .rule-name { font-weight: 700; }
  .expand { margin-left: 6px; font-size: 12px; }

  .stream {
    display: flex; flex-wrap: wrap; gap: 3px;
    overflow-y: auto; min-height: 0; flex: 1 1 auto;
    align-content: flex-start;
    padding: 8px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm);
  }

  .chip {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1;
    padding: 4px 6px;
    border: 1px solid;
    border-radius: 4px;
    white-space: pre;
    user-select: none;
  }
  .rulechip { font-weight: 700; }
  .stream-chip { transition: outline 0.1s, transform 0.1s; }
  .hot {
    outline: 2px solid var(--chosen);
    outline-offset: 1px;
    z-index: 1;
  }
  .next-note { font-size: 12.5px; margin: 8px 0 0; }
</style>
