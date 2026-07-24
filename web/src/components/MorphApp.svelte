<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { trace } from '../lib/mdl/engine';
  import type { Step } from '../lib/mdl/types';
  import {
    morphologyProblem,
    defaultConfig,
    type MorphModel,
    type MergeMove,
    type CodeMode
  } from '../lib/morphology/morphology';

  import WordListView from './morph/WordListView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  // The shell supplies the wordmark + lens switcher; the lens owns everything else.
  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('morph', [
    { id: 'words', title: 'Lexicon & segmentation' },
    { id: 'cands', title: 'Candidate merges' },
    { id: 'input', title: 'Word list (data)' },
    { id: 'cost', title: 'Description length' },
    { id: 'chart', title: 'Evolution' }
  ]);

  // Word lists chosen so shared stems + affixes pay off under MDL. "word n" sets
  // a corpus frequency; bare words default to 1. Frequency drives the merges.
  const SAMPLES: Record<string, string> = {
    'verb inflection': `walk 6
walks 3
walking 8
walked 4
talk 5
talks 2
talking 7
talked 3
jump 2
jumps 1
jumping 3
jumped 2`,
    'un- prefix': `happy 5
unhappy 3
kind 4
unkind 2
clear 4
unclear 2
lock 3
unlock 4
fair 3
unfair 2`,
    plurals: `cat 6
cats 4
dog 7
dogs 5
bird 3
birds 2
hand 4
hands 3
book 5
books 4`,
    'agreement (Spanish-ish)': `gato 4
gatos 3
gata 2
gatas 1
perro 5
perros 4
perra 2
perras 1
nino 3
ninos 2
nina 2
ninas 1`,
    'frequency matters': `the 100
then 4
there 5
they 8
them 6
a 80
at 7
an 9`
  };

  const DEFAULT_SAMPLE = 'verb inflection';
  let sampleKey = $state(DEFAULT_SAMPLE);
  let text = $state(SAMPLES[DEFAULT_SAMPLE]);
  let codeMode = $state<CodeMode>('uniform');
  let includeOverhead = $state(true);

  const player = new Player<Step<MorphModel, MergeMove>>();

  $effect(() => {
    const config = { ...defaultConfig, codeMode, includeOverhead };
    const problem = morphologyProblem(text.trim() || 'a', config);
    player.load(trace(problem, { maxSteps: 300 }));
  });

  function pickSample(k: string) {
    sampleKey = k;
    text = SAMPLES[k];
  }

  const cur = $derived(player.current);
  const reference = $derived(player.steps[0]?.cost.total ?? 1);
  const wordCount = $derived(cur?.model.words.length ?? 0);
</script>

<div class="topbar panel">
  {@render brand()}

  <span class="formula mono" title="minimize total bits = lexicon + segmented corpus">
    <b style="color:var(--total)">min</b>
    <b style="color:var(--model)">L(M)</b>+<b style="color:var(--data)">L(D|M)</b>
  </span>

  <span class="spacer"></span>

  <div class="f">
    <span class="lbl">code</span>
    <div class="toggle-group">
      <button class:active={codeMode === 'uniform'} onclick={() => (codeMode = 'uniform')}>log₂V</button>
      <button class:active={codeMode === 'shannon'} onclick={() => (codeMode = 'shannon')}>−log₂p</button>
    </div>
  </div>

  <label class="cb" title="count the bits to transmit the model-of-the-model (code table / lexicon framing)">
    <input type="checkbox" bind:checked={includeOverhead} /> overhead
  </label>

  <span class="chars mono muted">{wordCount}w</span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if cur}
  <PanelHost manager={panels}>
    <div class="col col-left">
      <Panel manager={panels} id="words" weight={1.15}>
        {#snippet actions()}
          <span class="mono">{cur.model.rules.length} morph{cur.model.rules.length === 1 ? '' : 's'}</span>
        {/snippet}
        <WordListView model={cur.model} chosen={cur.chosen} />
      </Panel>

      <Panel manager={panels} id="cands" weight={0.85}>
        {#snippet actions()}
          <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
        {/snippet}
        <CandidatesTable candidates={cur.candidates} baseline={cur.cost} chosen={cur.chosen} />
      </Panel>
    </div>

    <div class="col col-right">
      <Panel manager={panels} id="input" weight={0.85}>
        {#snippet actions()}
          <span class="mono">{wordCount} word{wordCount === 1 ? '' : 's'}</span>
        {/snippet}
        <div class="editor">
          <label class="f">
            <span class="lbl">sample</span>
            <select value={sampleKey} onchange={(e) => pickSample((e.currentTarget as HTMLSelectElement).value)}>
              {#each Object.keys(SAMPLES) as k}<option value={k}>{k}</option>{/each}
            </select>
          </label>
          <textarea class="data-input mono scrollbar" bind:value={text} spellcheck="false"
                    placeholder="one word per line, optional “word count”…"></textarea>
          <p class="hint faint">One word per line; add a count like <span class="mono">walking 8</span> to weight by frequency.</p>
        </div>
      </Panel>

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
  .spacer { flex: 1 1 auto; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .toggle-group button { padding: 4px 9px; font-size: 12px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; cursor: pointer; }
  .cb input { accent-color: var(--model); }
  .chars { font-size: 11px; white-space: nowrap; }
  .reset-layout { padding: 4px 9px; font-size: 11px; white-space: nowrap; }

  /* Word-list editor lives in its own panel now; the textarea fills it. */
  .editor { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  .editor .f { flex: 0 0 auto; }
  .data-input {
    flex: 1 1 auto; min-height: 60px; width: 100%; box-sizing: border-box;
    font-size: 12px; padding: 7px 9px; line-height: 1.5; resize: none; white-space: pre;
  }
  .hint { flex: 0 0 auto; font-size: 11px; margin: 0; line-height: 1.4; }

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
