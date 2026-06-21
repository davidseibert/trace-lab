<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { trainTrace, makeForward, type LlmStep } from '../lib/llm/trainTrace';
  import type { ForwardViz } from '../lib/llm/model';
  import { DATASETS, DATASET_KEYS } from '../lib/llm/datasets';
  import { tokenColor } from '../lib/llm/colors';

  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import ActGrid from './llm/ActGrid.svelte';
  import AttentionView from './llm/AttentionView.svelte';
  import OutputBars from './llm/OutputBars.svelte';
  import LossChart from './llm/LossChart.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('llm', [
    { id: 'tokens', title: 'Tokens' },
    { id: 'embed', title: 'Token embeddings' },
    { id: 'pos', title: 'With position' },
    { id: 'attn', title: 'Attention' },
    { id: 'ffn', title: 'Feed-forward' },
    { id: 'output', title: 'Next-token guess' },
    { id: 'loss', title: 'Loss over training' },
    { id: 'explain', title: 'Readout' }
  ]);

  const STEPS = 300;
  let datasetKey = $state(DATASET_KEYS[0]);
  let seed = $state(1);

  const ds = $derived(DATASETS[datasetKey]);
  const player = new Player<LlmStep>();

  // The fixed-probe trace + a forward function over the per-step weights, both
  // rebuilt only when the dataset or seed changes — never on playback.
  let forwardFn = $state<((stepIndex: number, tokenIds: number[]) => ForwardViz) | null>(null);

  $effect(() => {
    const run = trainTrace(DATASETS[datasetKey], { steps: STEPS, seed });
    player.load(run.steps);
    forwardFn = makeForward(run);
  });

  // --- Query box: replay an arbitrary prompt against the current step's model.
  let queryText = $state(DATASETS[DATASET_KEYS[0]].probe.join(' '));
  let committedQuery = $state<number[] | null>(null);

  // Reset the query whenever the dataset (and thus the vocabulary) changes.
  $effect(() => {
    queryText = DATASETS[datasetKey].probe.join(' ');
    committedQuery = null;
  });

  function parseQuery(text: string): { ids?: number[]; error?: string } {
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

  const parsed = $derived(parseQuery(queryText));
  const parseError = $derived(queryText.trim() === '' ? '' : (parsed.error ?? ''));

  function runQuery() {
    if (parsed.ids) committedQuery = parsed.ids;
  }
  function clearQuery() {
    committedQuery = null;
    queryText = ds.probe.join(' ');
  }

  const cur = $derived(player.current);
  const isQuery = $derived(committedQuery !== null);

  // The forward pass currently on display: the user's committed query (replayed
  // at the current step), otherwise the training probe.
  const activeViz = $derived.by<ForwardViz | null>(() => {
    if (committedQuery && forwardFn) return forwardFn(player.index, committedQuery);
    return cur?.viz ?? null;
  });

  const activeTokens = $derived(activeViz ? activeViz.tokenIds.map((id) => ds.vocab[id]) : []);
  const activeColors = $derived(activeTokens.map((t) => tokenColor(ds, t)));
  const vocabColors = $derived(ds.vocab.map((t) => tokenColor(ds, t)));

  const activePred = $derived.by(() => {
    if (!activeViz) return { id: 0, token: '', conf: 0 };
    let id = 0;
    for (let v = 1; v < activeViz.probs.length; v++) if (activeViz.probs[v] > activeViz.probs[id]) id = v;
    return { id, token: ds.vocab[id], conf: activeViz.probs[id] };
  });

  const stepLoss = $derived(cur?.loss ?? 0);
  const note = $derived(cur?.note ?? '');
  const probeLearned = $derived(cur ? cur.predToken === ds.probeTarget : false);
</script>

<div class="topbar panel">
  {@render brand()}

  <label class="f">
    <span class="lbl">dataset</span>
    <select value={datasetKey} onchange={(e) => (datasetKey = (e.currentTarget as HTMLSelectElement).value)}>
      {#each DATASET_KEYS as k}<option value={k}>{DATASETS[k].label}</option>{/each}
    </select>
  </label>

  <div class="f query">
    <span class="lbl">prompt</span>
    <input
      class="q-input mono"
      class:invalid={!!parseError}
      bind:value={queryText}
      onkeydown={(e) => e.key === 'Enter' && runQuery()}
      spellcheck="false"
      placeholder={ds.probe.join(' ')}
      title={`vocabulary: ${ds.vocab.join(' ')}`}
    />
    <button class="ghost" onclick={runQuery} disabled={!parsed.ids}>▶ run</button>
    {#if isQuery}
      <button class="ghost" onclick={clearQuery} title="Back to the training probe">✕ probe</button>
    {/if}
  </div>

  {#if parseError}
    <span class="qerr mono" title={`vocabulary: ${ds.vocab.join(' ')}`}>{parseError}</span>
  {:else}
    <span class="viewing mono" class:q={isQuery}>{isQuery ? 'your prompt' : 'training probe'}</span>
  {/if}

  <button class="ghost" title="Re-initialise weights with a new random seed" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  <span class="spacer"></span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if activeViz}
  <PanelHost manager={panels}>
    <div class="col col-a">
      <Panel manager={panels} id="tokens" fit>
        {#snippet actions()}<span class="mono">{activeTokens.length} tokens · {isQuery ? 'query' : 'probe'}</span>{/snippet}
        <div class="tokens">
          {#each activeTokens as t, i}
            <span class="tchip mono" style="color:{activeColors[i]}; border-color:{activeColors[i]}">
              <span class="tpos faint">{i}</span>{t}
            </span>
          {/each}
        </div>
      </Panel>

      <Panel manager={panels} id="embed" weight={1}>
        {#snippet actions()}<span class="mono">{ds.embDim}d</span>{/snippet}
        <ActGrid matrix={activeViz.tokEmb} rowLabels={activeTokens} rowColors={activeColors} signed />
      </Panel>

      <Panel manager={panels} id="pos" weight={1}>
        {#snippet actions()}<span class="mono">token + position</span>{/snippet}
        <ActGrid matrix={activeViz.embedded} rowLabels={activeTokens} rowColors={activeColors} signed />
      </Panel>
    </div>

    <div class="col col-b">
      <Panel manager={panels} id="attn" weight={1.5}>
        {#snippet actions()}<span class="mono">{ds.nHeads} heads</span>{/snippet}
        <AttentionView attn={activeViz.attn} tokens={activeTokens} tokenColors={activeColors} />
      </Panel>

      <Panel manager={panels} id="ffn" weight={1}>
        {#snippet actions()}<span class="mono">{ds.ffnHid} units</span>{/snippet}
        <ActGrid matrix={activeViz.ffnHidden} rowLabels={activeTokens} rowColors={activeColors} signed={false} colLabel="units" />
      </Panel>
    </div>

    <div class="col col-c">
      <Panel manager={panels} id="output" weight={1.3}>
        {#snippet actions()}<span class="mono">{(activePred.conf * 100).toFixed(0)}% peak</span>{/snippet}
        <OutputBars probs={activeViz.probs} vocab={ds.vocab} colors={vocabColors} predId={activePred.id} targetToken={isQuery ? '' : ds.probeTarget} />
      </Panel>

      <Panel manager={panels} id="loss" weight={1}>
        {#snippet actions()}<span class="mono">loss {stepLoss.toFixed(3)}</span>{/snippet}
        <LossChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
      </Panel>

      <Panel manager={panels} id="explain" fit>
        <div class="readout">
          {#if isQuery}
            <p class="say">
              Your prompt <span class="mono">“{activeTokens.join(' ')}”</span> at step <b>{player.index}</b>
              → the model predicts
              <b class="mono" style="color:{vocabColors[activePred.id]}">{activePred.token}</b>
              at <b>{(activePred.conf * 100).toFixed(0)}%</b>.
            </p>
            <p class="say muted">Scrub the timeline to watch this prediction form as the model trains.</p>
          {:else}
            <p class="say">
              After <b>{player.index}</b> step{player.index === 1 ? '' : 's'}, the model reads
              <span class="mono">“{ds.probe.join(' ')}”</span> and predicts
              <b class="mono" style="color:{vocabColors[activePred.id]}">{activePred.token}</b>
              at <b>{(activePred.conf * 100).toFixed(0)}%</b>.
            </p>
            <p class="say muted">
              Correct continuation <span class="mono" style="color:{tokenColor(ds, ds.probeTarget)}">{ds.probeTarget}</span>
              sits at <b>{(cur ? cur.targetProb * 100 : 0).toFixed(0)}%</b>
              {#if probeLearned}<span class="good"> — learned ✓</span>{/if}
            </p>
          {/if}
        </div>
      </Panel>
    </div>
  </PanelHost>

  <div class="panel transport-panel">
    <Controls {player} />
    <div class="note mono" class:converged={!cur?.chosen} title={note}>{note}</div>
  </div>
{/if}

<style>
  .topbar {
    display: flex;
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    padding: 7px 12px;
  }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .query { flex: 1; min-width: 220px; }
  .q-input { flex: 1; min-width: 120px; font-size: 12px; padding: 5px 8px; }
  .q-input.invalid { border-color: var(--bad); }
  .qerr { font-size: 11px; color: var(--bad); white-space: nowrap; }
  .viewing { font-size: 11px; color: var(--muted); white-space: nowrap; }
  .viewing.q { color: var(--chosen); }
  .spacer { flex: 0 1 auto; margin-left: auto; }
  .reset-layout { padding: 4px 9px; font-size: 11px; white-space: nowrap; }

  .col { display: flex; flex-direction: column; gap: 8px; min-height: 0; min-width: 0; }
  .col-a { flex: 1 1 0; }
  .col-b { flex: 1.3 1 0; }
  .col-c { flex: 1.1 1 0; }

  .tokens { display: flex; flex-wrap: wrap; gap: 5px; }
  .tchip {
    display: inline-flex; align-items: baseline; gap: 4px;
    font-size: 13px; padding: 3px 7px;
    border: 1px solid; border-radius: 5px; background: var(--bg-2);
  }
  .tpos { font-size: 9px; }

  .readout { display: flex; flex-direction: column; gap: 6px; }
  .say { margin: 0; font-size: 12.5px; line-height: 1.45; }

  .transport-panel {
    display: flex;
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 16px;
    padding: 7px 12px;
  }
  .note {
    flex: 1; min-width: 0;
    font-size: 12px; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    text-align: right;
  }
  .note.converged { color: var(--good); }
</style>
