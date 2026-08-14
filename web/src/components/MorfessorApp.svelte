<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import {
    morfessorTrace,
    buildWords,
    alphabetOf,
    defaultConfig,
    type MorfStep
  } from '../lib/morfessor/morfessor';
  import { MORPH_SAMPLES, MORPH_DEFAULT_SAMPLE } from '../lib/morphology/samples';
  import { LensSettings } from '../lib/sampleState.svelte';

  import CorpusView from './morf/CorpusView.svelte';
  import ReanalysisView from './morf/ReanalysisView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import CodeControls from './concept/CodeControls.svelte';
  import WordListEditor from './concept/WordListEditor.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  const panels = new PanelManager(
    'morfessor',
    [
      { id: 'state', title: 'Lexicon & corpus' },
      { id: 'reanalysis', title: 'Re-analysis' },
      { id: 'input', title: 'Word list (data)' },
      { id: 'cost', title: 'Description length', fit: true },
      { id: 'chart', title: 'Evolution' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['state', 'reanalysis'], ['input', 'cost', 'chart', 'guide']],
      widths: [1.45, 1],
      weights: { state: 1.3, reanalysis: 0.9, input: 0.85 }
    }
  );

  const SAMPLES = MORPH_SAMPLES;

  // This lens defaults to the Shannon code and has no overhead toggle.
  const settings = new LensSettings({
    samples: SAMPLES,
    defaultSample: MORPH_DEFAULT_SAMPLE,
    defaultCodeMode: 'shannon',
    overhead: false
  });

  const player = new Player<MorfStep>();

  $effect(() => {
    const words = buildWords(settings.text.trim() || 'a');
    const A = alphabetOf(words);
    const config = { ...defaultConfig, codeMode: settings.codeMode };
    player.load(morfessorTrace(words, A, config, { maxSteps: 2000 }));
  });

  const cur = $derived(player.current);
  const reference = $derived(player.steps[0]?.cost.total ?? 1);
  const wordCount = $derived(cur?.model.words.length ?? 0);
  const epochLabel = $derived(!cur ? '' : cur.epoch < 0 ? 'converged' : `epoch ${cur.epoch + 1}`);
</script>

<TopBar {panels}>
  <CodeControls
    bind:codeMode={settings.codeMode}
    formulaTitle="minimize total bits = lexicon + segmented corpus"
    overhead={false}
  >
    <span class="spacer"></span>
  </CodeControls>

  <span class="chars mono muted">{wordCount}w</span>
</TopBar>

{#if cur}
  {#snippet aState()}
    <span class="mono">{epochLabel}</span>
  {/snippet}
  {#snippet pState()}
    <CorpusView model={cur.model} focusWord={cur.focusWord} />
  {/snippet}

  {#snippet aReanalysis()}
    <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pReanalysis()}
    <ReanalysisView step={cur} />
  {/snippet}

  {#snippet aInput()}
    <span class="mono">{wordCount} word{wordCount === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pInput()}
    <WordListEditor bind:text={settings.text} bind:sampleKey={settings.sampleKey} samples={SAMPLES}>
      {#snippet hint()}Every word starts whole; the sweep splits it to share morphs. Add a count like <span class="mono">walking 8</span> to weight by frequency.{/snippet}
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
    <InterpretGuide lens="morfessor" sections={['mdlcore', 'codes', 'morphpair']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ state: pState, reanalysis: pReanalysis, input: pInput, cost: pCost, chart: pChart, guide: pGuide }}
    actions={{ state: aState, reanalysis: aReanalysis, input: aInput, cost: aCost }}
  />

  <TransportBar {player} note={cur.note} converged={cur.epoch < 0} />
{/if}
