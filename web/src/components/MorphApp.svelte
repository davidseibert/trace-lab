<script lang="ts">
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
  import { router } from '../lib/router.svelte';
  import { MORPH_SAMPLES, MORPH_DEFAULT_SAMPLE } from '../lib/morphology/samples';

  import WordListView from './morph/WordListView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
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
  const DEFAULT_SAMPLE = MORPH_DEFAULT_SAMPLE;

  const initialSample =
    router.get('sample') && SAMPLES[router.get('sample')!] ? router.get('sample')! : DEFAULT_SAMPLE;
  let sampleKey = $state(initialSample);
  let text = $state(router.get('text') ?? SAMPLES[initialSample]);
  let codeMode = $state<CodeMode>(router.get('code') === 'shannon' ? 'shannon' : 'uniform');
  let includeOverhead = $state(router.bool('oh') ?? true);

  $effect(() => {
    router.setQuery({
      sample: sampleKey === DEFAULT_SAMPLE ? null : sampleKey,
      text: text === SAMPLES[sampleKey] ? null : text,
      code: codeMode === 'uniform' ? null : codeMode,
      oh: includeOverhead ? null : false
    });
  });

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

<TopBar {panels}>
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
    <div class="editor">
          <label class="f">
            <span class="lbl">sample</span>
            <select value={sampleKey} onchange={(e) => pickSample((e.currentTarget as HTMLSelectElement).value)}>
              {#each Object.keys(SAMPLES) as k}<option value={k}>{k}</option>{/each}
            </select>
          </label>
          <textarea class="word-input mono scrollbar" bind:value={text} spellcheck="false"
                    placeholder="one word per line, optional “word count”…"></textarea>
          <p class="editor-hint faint">One word per line; add a count like <span class="mono">walking 8</span> to weight by frequency.</p>
        </div>
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

<style>
  /* Word-list editor lives in its own panel; the textarea fills it. */
  .editor { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  .editor .f { flex: 0 0 auto; }
  .word-input {
    flex: 1 1 auto; min-height: 60px; width: 100%; box-sizing: border-box;
    font-size: 12px; padding: 7px 9px; line-height: 1.5; resize: none; white-space: pre;
  }
  .editor-hint { flex: 0 0 auto; font-size: 11px; margin: 0; line-height: 1.4; white-space: normal; }
</style>
