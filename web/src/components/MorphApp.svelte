<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import { trace } from '../lib/mdl/engine';
  import type { Step } from '../lib/mdl/types';
  import {
    morphologyProblem,
    defaultConfig,
    type MorphModel,
    type MergeMove
  } from '../lib/morphology/morphology';
  import { MORPH_SAMPLES, MORPH_DEFAULT_SAMPLE } from '../lib/morphology/samples';
  import { LensSettings } from '../lib/sampleState.svelte';

  import WordListView from './morph/WordListView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import CodeControls from './concept/CodeControls.svelte';
  import WordListEditor from './concept/WordListEditor.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  const panels = new PanelManager(
    'morph',
    [
      { id: 'words', title: 'Lexicon & segmentation', zoomable: true },
      { id: 'cands', title: 'Candidate merges', zoomable: true },
      { id: 'input', title: 'Word list (data)' },
      { id: 'cost', title: 'Description length', fit: true },
      { id: 'chart', title: 'Evolution' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['words', 'cands'], ['input', 'cost', 'chart', 'guide']],
      widths: [1.45, 1],
      weights: { words: 1.15, cands: 0.85, input: 0.85 }
    }
  );

  const SAMPLES = MORPH_SAMPLES;

  const settings = new LensSettings({ samples: SAMPLES, defaultSample: MORPH_DEFAULT_SAMPLE });

  const player = new Player<Step<MorphModel, MergeMove>>();

  $effect(() => {
    const config = {
      ...defaultConfig,
      codeMode: settings.codeMode,
      includeOverhead: settings.includeOverhead
    };
    const problem = morphologyProblem(settings.text.trim() || 'a', config);
    player.load(trace(problem, { maxSteps: 300 }));
  });

  const cur = $derived(player.current);
  const reference = $derived(player.steps[0]?.cost.total ?? 1);
  const wordCount = $derived(cur?.model.words.length ?? 0);
</script>

<TopBar {panels}>
  <CodeControls
    bind:codeMode={settings.codeMode}
    bind:includeOverhead={settings.includeOverhead}
    formulaTitle="minimize total bits = lexicon + segmented corpus"
    overheadTitle="count the bits to transmit the model-of-the-model (code table / lexicon framing)"
  >
    <span class="spacer"></span>
  </CodeControls>

  <span class="chars mono muted">{wordCount}w</span>
</TopBar>

{#if cur}
  {#snippet aWords()}
    <span class="mono">{cur.model.rules.length} morph{cur.model.rules.length === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pWords()}
    <WordListView model={cur.model} chosen={cur.chosen} />
  {/snippet}

  {#snippet aCands()}
    <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pCands()}
    <CandidatesTable candidates={cur.candidates} baseline={cur.cost} chosen={cur.chosen} />
  {/snippet}

  {#snippet aInput()}
    <span class="mono">{wordCount} word{wordCount === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pInput()}
    <WordListEditor bind:text={settings.text} bind:sampleKey={settings.sampleKey} samples={SAMPLES}>
      {#snippet hint()}One word per line; add a count like <span class="mono">walking 8</span> to weight by frequency.{/snippet}
    </WordListEditor>
  {/snippet}

  {#snippet aCost()}
    <span class="mono">step {player.index}</span>
  {/snippet}
  {#snippet pCost()}
    <CostPanel cost={cur.cost} {reference} />
  {/snippet}

  {#snippet pChart()}
    <CostChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="morph" sections={['mdlcore', 'codes', 'morphpair']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ words: pWords, cands: pCands, input: pInput, cost: pCost, chart: pChart, guide: pGuide }}
    actions={{ words: aWords, cands: aCands, input: aInput, cost: aCost }}
  />

  <TransportBar {player} note={cur.note} converged={!cur.chosen} />
{/if}
