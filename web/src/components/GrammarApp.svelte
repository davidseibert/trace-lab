<script lang="ts">
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
  import { router } from '../lib/router.svelte';

  import StreamView from './StreamView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  const panels = new PanelManager(
    'grammar',
    [
      { id: 'grammar', title: 'Grammar' },
      { id: 'cands', title: 'Candidate moves', zoomable: true },
      { id: 'cost', title: 'Description length', fit: true },
      { id: 'chart', title: 'Evolution' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['grammar', 'cands'], ['cost', 'chart', 'guide']],
      widths: [1.45, 1],
      weights: { grammar: 1.15, cands: 0.85 }
    }
  );

  const SAMPLES: Record<string, string> = {
    'the cat / the mat': 'the cat sat on the mat. the cat ate the rat. the rat sat.',
    'to be or not to be': 'to be or not to be, that is the question',
    'nested: theatre': 'the theme of the theatre is the theory of the theatre',
    'pure repetition': 'abcabcabcabcabcabc',
    'DNA-ish': 'ACGTACGTTACGACGTACGTTACG',
    'mississippi': 'mississippi river, mississippi delta, miss the mississippi'
  };

  const DEFAULT_SAMPLE = 'the cat / the mat';
  const initialSample =
    router.get('sample') && SAMPLES[router.get('sample')!] ? router.get('sample')! : DEFAULT_SAMPLE;
  let sampleKey = $state(initialSample);
  let text = $state(router.get('text') ?? SAMPLES[initialSample]);
  let codeMode = $state<CodeMode>(router.get('code') === 'shannon' ? 'shannon' : 'uniform');
  let includeOverhead = $state(router.bool('oh') ?? true);

  // Settings live in the URL: refresh keeps them, and the URL is shareable.
  $effect(() => {
    router.setQuery({
      sample: sampleKey === DEFAULT_SAMPLE ? null : sampleKey,
      text: text === SAMPLES[sampleKey] ? null : text,
      code: codeMode === 'uniform' ? null : codeMode,
      oh: includeOverhead ? null : false
    });
  });

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

<TopBar {panels}>
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
</TopBar>

{#if cur}
  {#snippet aGrammar()}
    <span class="mono">{cur.model.rules.length} rule{cur.model.rules.length === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pGrammar()}
    <StreamView model={cur.model} chosen={cur.chosen} />
  {/snippet}

  {#snippet aCands()}
    <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
  {/snippet}
  {#snippet pCands()}
    <CandidatesTable candidates={cur.candidates} baseline={cur.cost} chosen={cur.chosen} />
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
    <InterpretGuide lens="grammar" sections={['mdlcore', 'codes']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ grammar: pGrammar, cands: pCands, cost: pCost, chart: pChart, guide: pGuide }}
    actions={{ grammar: aGrammar, cands: aCands, cost: aCost }}
  />

  <TransportBar {player} note={cur.note} converged={!cur.chosen} />
{/if}
