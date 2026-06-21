<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { trainTrace, type LlmStep } from '../lib/llm/trainTrace';
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

  // Recompute the whole training trace only when the dataset or seed changes —
  // never on playback. Same (dataset, seed) ⇒ identical trace.
  $effect(() => {
    player.load(trainTrace(DATASETS[datasetKey], { steps: STEPS, seed }));
  });

  const cur = $derived(player.current);

  // Token strings + colors for the current probe input.
  const tokens = $derived(cur ? cur.viz.tokenIds.map((id) => ds.vocab[id]) : []);
  const tokColors = $derived(tokens.map((t) => tokenColor(ds, t)));
  const vocabColors = $derived(ds.vocab.map((t) => tokenColor(ds, t)));
  const learned = $derived(cur ? cur.predToken === ds.probeTarget : false);
</script>

<div class="topbar panel">
  {@render brand()}

  <span class="subtitle">{ds.subtitle}</span>

  <label class="f">
    <span class="lbl">dataset</span>
    <select value={datasetKey} onchange={(e) => (datasetKey = (e.currentTarget as HTMLSelectElement).value)}>
      {#each DATASET_KEYS as k}<option value={k}>{DATASETS[k].label}</option>{/each}
    </select>
  </label>

  <div class="f">
    <span class="lbl">probe</span>
    <span class="probe mono">{ds.probe.join(' ')} <span class="faint">→ {ds.probeTarget}</span></span>
  </div>

  <button class="ghost" title="Re-initialise weights with a new random seed" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  <span class="spacer"></span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if cur}
  <PanelHost manager={panels}>
    <div class="col col-a">
      <Panel manager={panels} id="tokens" fit>
        {#snippet actions()}<span class="mono">{tokens.length} tokens</span>{/snippet}
        <div class="tokens">
          {#each tokens as t, i}
            <span class="tchip mono" style="color:{tokColors[i]}; border-color:{tokColors[i]}">
              <span class="tpos faint">{i}</span>{t}
            </span>
          {/each}
        </div>
      </Panel>

      <Panel manager={panels} id="embed" weight={1}>
        {#snippet actions()}<span class="mono">{ds.embDim}d</span>{/snippet}
        <ActGrid matrix={cur.viz.tokEmb} rowLabels={tokens} rowColors={tokColors} signed />
      </Panel>

      <Panel manager={panels} id="pos" weight={1}>
        {#snippet actions()}<span class="mono">token + position</span>{/snippet}
        <ActGrid matrix={cur.viz.embedded} rowLabels={tokens} rowColors={tokColors} signed />
      </Panel>
    </div>

    <div class="col col-b">
      <Panel manager={panels} id="attn" weight={1.5}>
        {#snippet actions()}<span class="mono">{ds.nHeads} heads</span>{/snippet}
        <AttentionView attn={cur.viz.attn} {tokens} tokenColors={tokColors} />
      </Panel>

      <Panel manager={panels} id="ffn" weight={1}>
        {#snippet actions()}<span class="mono">{ds.ffnHid} units</span>{/snippet}
        <ActGrid matrix={cur.viz.ffnHidden} rowLabels={tokens} rowColors={tokColors} signed={false} colLabel="units" />
      </Panel>
    </div>

    <div class="col col-c">
      <Panel manager={panels} id="output" weight={1.3}>
        {#snippet actions()}<span class="mono">{(cur.confidence * 100).toFixed(0)}% peak</span>{/snippet}
        <OutputBars probs={cur.viz.probs} vocab={ds.vocab} colors={vocabColors} predId={cur.predId} targetToken={ds.probeTarget} />
      </Panel>

      <Panel manager={panels} id="loss" weight={1}>
        {#snippet actions()}<span class="mono">loss {cur.loss.toFixed(3)}</span>{/snippet}
        <LossChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
      </Panel>

      <Panel manager={panels} id="explain" fit>
        <div class="readout">
          <p class="say">
            After <b>{player.index}</b> step{player.index === 1 ? '' : 's'}, the model reads
            <span class="mono">“{ds.probe.join(' ')}”</span> and predicts
            <b class="mono" style="color:{tokColors.length ? vocabColors[cur.predId] : 'var(--text)'}">{cur.predToken}</b>
            at <b>{(cur.confidence * 100).toFixed(0)}%</b>.
          </p>
          <p class="say muted">
            Correct continuation <span class="mono" style="color:{tokenColor(ds, ds.probeTarget)}">{ds.probeTarget}</span>
            sits at <b>{(cur.targetProb * 100).toFixed(0)}%</b>
            {#if learned}<span class="good"> — learned ✓</span>{/if}
          </p>
        </div>
      </Panel>
    </div>
  </PanelHost>

  <div class="panel transport-panel">
    <Controls {player} />
    <div class="note mono" class:converged={!cur.chosen} title={cur.note}>{cur.note}</div>
  </div>
{/if}

<style>
  .topbar {
    display: flex;
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 14px;
    padding: 7px 12px;
  }
  .subtitle { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .probe { font-size: 12px; white-space: nowrap; }
  .spacer { flex: 1; }
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
