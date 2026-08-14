<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import {
    morfessorTrace,
    buildWords,
    alphabetOf,
    defaultConfig,
    type MorfStep,
    type CodeMode
  } from '../lib/morfessor/morfessor';
  import { router } from '../lib/router.svelte';
  import { MORPH_SAMPLES, MORPH_DEFAULT_SAMPLE } from '../lib/morphology/samples';

  import CorpusView from './morf/CorpusView.svelte';
  import ReanalysisView from './morf/ReanalysisView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import InterpretGuide from './InterpretGuide.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  const panels = new PanelManager('morfessor', [
    { id: 'state', title: 'Lexicon & corpus' },
    { id: 'reanalysis', title: 'Re-analysis' },
    { id: 'input', title: 'Word list (data)' },
    { id: 'cost', title: 'Description length' },
    { id: 'chart', title: 'Evolution' },
    { id: 'guide', title: 'How to read this', collapsed: true }
  ]);

  const SAMPLES = MORPH_SAMPLES;
  const DEFAULT_SAMPLE = MORPH_DEFAULT_SAMPLE;

  const initialSample =
    router.get('sample') && SAMPLES[router.get('sample')!] ? router.get('sample')! : DEFAULT_SAMPLE;
  let sampleKey = $state(initialSample);
  let text = $state(router.get('text') ?? SAMPLES[initialSample]);
  let codeMode = $state<CodeMode>(router.get('code') === 'uniform' ? 'uniform' : 'shannon');

  $effect(() => {
    router.setQuery({
      sample: sampleKey === DEFAULT_SAMPLE ? null : sampleKey,
      text: text === SAMPLES[sampleKey] ? null : text,
      code: codeMode === 'shannon' ? null : codeMode
    });
  });

  const player = new Player<MorfStep>();

  $effect(() => {
    const words = buildWords(text.trim() || 'a');
    const A = alphabetOf(words);
    const config = { ...defaultConfig, codeMode };
    player.load(morfessorTrace(words, A, config, { maxSteps: 2000 }));
  });

  function pickSample(k: string) {
    sampleKey = k;
    text = SAMPLES[k];
  }

  const cur = $derived(player.current);
  const reference = $derived(player.steps[0]?.cost.total ?? 1);
  const wordCount = $derived(cur?.model.words.length ?? 0);
  const epochLabel = $derived(!cur ? '' : cur.epoch < 0 ? 'converged' : `epoch ${cur.epoch + 1}`);
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

  <span class="chars mono muted">{wordCount}w</span>
</TopBar>

{#if cur}
  <PanelHost manager={panels}>
    <div class="col col-left">
      <Panel manager={panels} id="state" weight={1.3}>
        {#snippet actions()}
          <span class="mono">{epochLabel}</span>
        {/snippet}
        <CorpusView model={cur.model} focusWord={cur.focusWord} />
      </Panel>

      <Panel manager={panels} id="reanalysis" weight={0.9}>
        {#snippet actions()}
          <span class="mono">{cur.candidates.length} candidate{cur.candidates.length === 1 ? '' : 's'}</span>
        {/snippet}
        <ReanalysisView step={cur} />
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
          <textarea class="word-input mono scrollbar" bind:value={text} spellcheck="false"
                    placeholder="one word per line, optional “word count”…"></textarea>
          <p class="editor-hint faint">Every word starts whole; the sweep splits it to share morphs. Add a count like <span class="mono">walking 8</span> to weight by frequency.</p>
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

      <Panel manager={panels} id="guide" weight={1}>
        <InterpretGuide lens="morfessor" sections={['mdlcore', 'codes', 'morphpair']} />
      </Panel>
    </div>
  </PanelHost>

  <TransportBar {player} note={cur.note} converged={cur.epoch < 0} />
{/if}

<style>
  .col-left { flex: 1.45 1 0; }
  .col-right { flex: 1 1 0; }

  .editor { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  .editor .f { flex: 0 0 auto; }
  .word-input {
    flex: 1 1 auto; min-height: 60px; width: 100%; box-sizing: border-box;
    font-size: 12px; padding: 7px 9px; line-height: 1.5; resize: none; white-space: pre;
  }
  .editor-hint { flex: 0 0 auto; font-size: 11px; margin: 0; line-height: 1.4; white-space: normal; }
</style>
