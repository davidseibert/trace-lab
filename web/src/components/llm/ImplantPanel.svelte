<script lang="ts">
  import type { Dataset } from '../../lib/llm/datasets';
  import type { FactSpec } from '../../lib/llm/implant';
  import type { InterferenceRow } from '../../lib/llm/lab';
  import { tokenColor } from '../../lib/llm/colors';

  let {
    ds,
    facts,
    rows,
    onAdd,
    onRemove
  }: {
    ds: Dataset;
    facts: FactSpec[];
    /** Δbits table: every eval case before → after the implants. */
    rows: InterferenceRow[];
    onAdd: (fact: Omit<FactSpec, 'key'>) => void;
    onRemove: (key: string) => void;
  } = $props();

  let promptText = $state('');
  let targetTok = $state('');
  let strength = $state(3);
  let whiten = $state(true);
  let threshold = $state(0.75);

  // Reset the form when the dataset (and thus the vocabulary) changes.
  $effect(() => {
    promptText = ds.probe.join(' ');
    targetTok = ds.vocab[0];
  });

  function parsePrompt(text: string): { ids?: number[]; error?: string } {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length === 0) return { error: 'type a prompt' };
    if (words.length > ds.ctxLen) return { error: `too long — max ${ds.ctxLen} tokens` };
    const ids: number[] = [];
    for (const w of words) {
      const id = ds.vocab.indexOf(w);
      if (id === -1) return { error: `unknown token “${w}”` };
      ids.push(id);
    }
    return { ids };
  }

  const parsed = $derived(parsePrompt(promptText));

  function add() {
    if (!parsed.ids) return;
    onAdd({
      promptIds: parsed.ids,
      targetId: ds.vocab.indexOf(targetTok),
      strength,
      whiten,
      threshold
    });
  }

  const promptOf = (f: FactSpec) => f.promptIds.map((id) => ds.vocab[id]).join(' ');

  const deltaClass = (d: number) => (d < -0.2 ? 'good' : d > 0.2 ? 'bad' : 'flat');
</script>

<div class="implant scrollbar">
  <div class="form">
    <div class="frow">
      <input
        class="p-input mono"
        class:invalid={!!parsed.error}
        bind:value={promptText}
        spellcheck="false"
        title={`key context — vocabulary: ${ds.vocab.join(' ')}`}
        onkeydown={(e) => e.key === 'Enter' && add()}
      />
      <span class="arrow mono faint">→</span>
      <select bind:value={targetTok} title="the token this key should recall">
        {#each ds.vocab as t (t)}<option value={t}>{t}</option>{/each}
      </select>
      <button class="ghost" onclick={add} disabled={!parsed.ids} title="Graft one FFN unit: key = the prompt's residual, value = the target's unembed direction. No training.">
        ⚡ implant
      </button>
    </div>
    <div class="frow knobs">
      <label class="k" title="target-logit boost when the memory fires (calibrated through the final LayerNorm: ×e^s odds for the target)">
        <span class="klbl">strength</span>
        <input type="range" min="0.5" max="8" step="0.5" bind:value={strength} />
        <span class="mono kval">{strength.toFixed(1)}</span>
      </label>
      <label class="k" title="the unit fires when input·key exceeds this fraction of the fact's own alignment — higher = more selective memory">
        <span class="klbl">gate θ</span>
        <input type="range" min="0.4" max="0.95" step="0.05" bind:value={threshold} />
        <span class="mono kval">{threshold.toFixed(2)}</span>
      </label>
      <label class="k chk" title="subtract the mean FFN-input (the shared / anisotropy component) from the key before storing — the associative-memory capacity condition. Often changes little here: ln2 already hands the FFN whitened keys.">
        <input type="checkbox" bind:checked={whiten} />
        <span class="klbl">whiten key</span>
      </label>
    </div>
    {#if parsed.error && promptText.trim()}
      <div class="perr mono">{parsed.error}</div>
    {/if}
  </div>

  {#if facts.length === 0}
    <p class="hint faint">
      No facts implanted. Write one straight into the FFN — a new hidden unit holding
      key → value as an outer product, Hebbian-style. Then scrub training and watch
      the same surgery behave differently against dumber weights.
    </p>
  {:else}
    <div class="facts">
      {#each facts as f (f.key)}
        <span class="fact mono">
          <span class="fp">{promptOf(f)}</span>
          <span class="faint">→</span>
          <span style="color:{tokenColor(ds, ds.vocab[f.targetId])}">{ds.vocab[f.targetId]}</span>
          <span class="fmeta faint">s{f.strength.toFixed(1)} θ{f.threshold.toFixed(2)}{f.whiten ? ' ⚪whitened' : ' ⚠raw'}</span>
          <button class="x" onclick={() => onRemove(f.key)} title="remove this implant">✕</button>
        </span>
      {/each}
    </div>

    <div class="table">
      <div class="trow thead mono faint">
        <span>sequence</span><span class="num">bits before</span><span class="num">after</span><span class="num">Δ</span>
      </div>
      {#each rows as r, i (i)}
        {@const d = r.after - r.before}
        <div class="trow mono" class:isfact={r.isFact} class:touched={r.touched}>
          <span class="tlabel" title={r.label + (r.touched ? ' — the fact’s own context: this is the overwrite, not interference' : '')}>
            {#if r.isFact}<span class="ow">★</span>{:else if r.touched}<span class="ow" title="overwritten by a fact">✎</span>{/if}{r.label}
          </span>
          <span class="num">{r.before.toFixed(2)}</span>
          <span class="num">{r.after.toFixed(2)}</span>
          <span class="num delta {deltaClass(d)}">{d >= 0 ? '+' : ''}{d.toFixed(2)}</span>
        </div>
      {/each}
      <div class="tfoot faint">
        Rows are description lengths −log₂ p in bits (whole sequences pay for every
        transition). Δ &lt; 0 on the ★ fact row = recall works. ✎ rows share the fact's
        context — that cost is the overwrite itself. Δ &gt; 0 anywhere else is
        interference: the key also fires on unrelated prompts. Scrub left and re-read
        the table — early-training keys are undifferentiated and interfere wildly.
        The whiten toggle often barely matters, and that's the lesson: the FFN reads
        its keys through LayerNorm, which already stripped the shared component — the
        capacity condition is built into the architecture.
      </div>
    </div>
  {/if}
</div>

<style>
  .implant { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; flex: 1 1 auto; }

  .form { display: flex; flex-direction: column; gap: 5px; }
  .frow { display: flex; align-items: center; gap: 6px; }
  .p-input { flex: 1; min-width: 80px; font-size: 11.5px; padding: 4px 7px; }
  .p-input.invalid { border-color: var(--bad); }
  .arrow { flex: 0 0 auto; }
  select { font-size: 11.5px; }
  .perr { font-size: 10.5px; color: var(--bad); }

  .knobs { flex-wrap: wrap; gap: 10px; }
  .k { display: flex; align-items: center; gap: 5px; }
  .k input[type='range'] { width: 74px; }
  .klbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
  .kval { font-size: 10.5px; min-width: 28px; }
  .chk { cursor: pointer; }

  .hint { font-size: 11px; line-height: 1.45; margin: 0; }

  .facts { display: flex; flex-direction: column; gap: 4px; }
  .fact {
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    font-size: 11px; padding: 3px 7px;
    border: 1px solid var(--border-2); border-radius: 5px; background: var(--bg-2);
  }
  .fp { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 55%; }
  .fmeta { font-size: 9.5px; }
  .x {
    margin-left: auto; border: none; background: none; color: var(--muted);
    cursor: pointer; font-size: 10px; padding: 0 2px;
  }
  .x:hover { color: var(--bad); }

  .table { display: flex; flex-direction: column; gap: 2px; }
  .trow {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 62px 46px 52px;
    gap: 6px; align-items: center;
    font-size: 10.5px; padding: 2px 5px; border-radius: 4px;
  }
  .thead { font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; }
  .trow.isfact { background: rgba(255, 209, 102, 0.08); outline: 1px solid rgba(255, 209, 102, 0.25); }
  .trow.touched { opacity: 0.85; }
  .ow { color: var(--chosen); margin-right: 4px; }
  .tlabel { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .num { text-align: right; }
  .delta.good { color: var(--good); }
  .delta.bad { color: var(--bad); }
  .delta.flat { color: var(--muted); }
  .tfoot { font-size: 9.5px; line-height: 1.4; padding: 3px 5px 0; }
</style>
