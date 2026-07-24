<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { trace } from '../lib/mdl/engine';
  import type { Step } from '../lib/mdl/types';
  import {
    grammarProblem,
    defaultConfig,
    type GrammarModel,
    type DigramMove,
    type CodeMode
  } from '../lib/string/grammar';

  import StreamView from './StreamView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  // The shell supplies the wordmark + lens switcher; the lens owns everything else.
  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('grammar', [
    { id: 'grammar', title: 'Grammar' },
    { id: 'cands', title: 'Candidate moves' },
    { id: 'cost', title: 'Description length' },
    { id: 'chart', title: 'Evolution' }
  ]);

  const SAMPLES: Record<string, string> = {
    'the cat / the mat': 'the cat sat on the mat. the cat ate the rat. the rat sat.',
    'to be or not to be': 'to be or not to be, that is the question',
    'nested: theatre': 'the theme of the theatre is the theory of the theatre',
    'pure repetition': 'abcabcabcabcabcabc',
    'DNA-ish': 'ACGTACGTTACGACGTACGTTACG',
    'mississippi': 'mississippi river, mississippi delta, miss the mississippi'
  };

  const DEFAULT_SAMPLE = 'the cat / the mat';
  let sampleKey = $state(DEFAULT_SAMPLE);
  let text = $state(SAMPLES[DEFAULT_SAMPLE]);
  let codeMode = $state<CodeMode>('uniform');
  let includeOverhead = $state(true);

  const player = new Player<Step<GrammarModel, DigramMove>>();

  $effect(() => {
    const config = { ...defaultConfig, codeMode, includeOverhead };
    const problem = grammarProblem(text || ' ', config);
    player.load(trace(problem, { maxSteps: 300 }));
  });

  function pickSample(k: string) {
    sampleKey = k;
    text = SAMPLES[k];
  }

  const cur = $derived(player.current);
  const reference = $derived(player.steps[0]?.cost.total ?? 1);
</script>

<div class="topbar panel">
  {@render brand()}

  <span class="formula mono" title="minimize total bits = model + data-given-model">
    <b style="color:var(--total)">min</b>
    <b style="color:var(--model)">L(M)</b>+<b style="color:var(--data)">L(D|M)</b>
  </span>

  <label class="f">
    <span class="lbl">dataset</span>
    <select value={sampleKey} onchange={(e) => pickSample((e.currentTarget as HTMLSelectElement).value)}>
      {#each Object.keys(SAMPLES) as k}<option value={k}>{k}</option>{/each}
    </select>
  </label>

  <input class="data-input mono" type="text" bind:value={text} spellcheck="false"
         placeholder="type any string…" />

  <div class="f">
    <span class="lbl">code</span>
    <div class="toggle-group">
      <button class:active={codeMode === 'uniform'} onclick={() => (codeMode = 'uniform')}>log₂V</button>
      <button class:active={codeMode === 'shannon'} onclick={() => (codeMode = 'shannon')}>−log₂p</button>
    </div>
  </div>

  <label class="cb" title="count the bits to transmit the model-of-the-model (code table / rule framing)">
    <input type="checkbox" bind:checked={includeOverhead} /> overhead
  </label>

  <span class="chars mono muted">{text.length}c</span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if cur}
  <PanelHost manager={panels}>
    <div class="col col-left">
      <Panel manager={panels} id="grammar" weight={1.15}>
        {#snippet actions()}
          <span class="mono">{cur.model.rules.length} rule{cur.model.rules.length === 1 ? '' : 's'}</span>
        {/snippet}
        <StreamView model={cur.model} chosen={cur.chosen} />
      </Panel>

      <Panel manager={panels} id="cands" weight={0.85}>
        {#snippet actions()}
          <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
        {/snippet}
        <CandidatesTable candidates={cur.candidates} baseline={cur.cost} chosen={cur.chosen} />
      </Panel>
    </div>

    <div class="col col-right">
      <Panel manager={panels} id="cost" fit>
        {#snippet actions()}
          <span class="mono">step {player.index}</span>
        {/snippet}
        <CostPanel cost={cur.cost} {reference} />
      </Panel>

      <Panel manager={panels} id="chart" weight={1}>
        <CostChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
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
  .formula { font-size: 12px; white-space: nowrap; }
  .formula b { font-weight: 700; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .data-input { flex: 1; min-width: 120px; font-size: 12px; padding: 5px 8px; }
  .toggle-group button { padding: 4px 9px; font-size: 12px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; cursor: pointer; }
  .cb input { accent-color: var(--model); }
  .chars { font-size: 11px; white-space: nowrap; }
  .reset-layout { padding: 4px 9px; font-size: 11px; white-space: nowrap; }

  .col { display: flex; flex-direction: column; gap: 8px; min-height: 0; min-width: 0; }
  .col-left { flex: 1.45 1 0; }
  .col-right { flex: 1 1 0; }

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
