<script lang="ts">
  import { Player } from './lib/player.svelte';
  import { trace } from './lib/mdl/engine';
  import {
    grammarProblem,
    defaultConfig,
    type GrammarModel,
    type DigramMove,
    type CodeMode
  } from './lib/string/grammar';

  import StreamView from './components/StreamView.svelte';
  import CostPanel from './components/CostPanel.svelte';
  import CostChart from './components/CostChart.svelte';
  import CandidatesTable from './components/CandidatesTable.svelte';
  import Controls from './components/Controls.svelte';

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

  const player = new Player<GrammarModel, DigramMove>();

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

<div class="app">
  <div class="topbar panel">
    <div class="brand">
      <h1>MDL<span class="dim">·Explorer</span></h1>
      <span class="formula mono" title="minimize total bits = model + data-given-model">
        <b style="color:var(--total)">min</b>
        <b style="color:var(--model)">L(M)</b>+<b style="color:var(--data)">L(D|M)</b>
      </span>
    </div>

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
  </div>

  {#if cur}
    <div class="body">
      <section class="panel stream-panel">
        <StreamView model={cur.model} chosen={cur.chosen} />
      </section>

      <section class="panel cands-panel">
        <CandidatesTable candidates={cur.candidates} baseline={cur.cost} chosen={cur.chosen} />
      </section>

      <aside class="side">
        <section class="panel cost-panel">
          <div class="panel-title row-title">
            <span>Description length</span>
            <span class="mono muted">step {player.index}</span>
          </div>
          <CostPanel cost={cur.cost} {reference} />
        </section>
        <section class="panel chart-panel">
          <CostChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
        </section>
      </aside>
    </div>

    <div class="panel transport-panel">
      <Controls {player} />
      <div class="note mono" class:converged={!cur.chosen} title={cur.note}>{cur.note}</div>
    </div>
  {/if}
</div>

<style>
  .app {
    height: 100dvh;
    max-width: 1500px;
    margin: 0 auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .panel { display: flex; flex-direction: column; min-height: 0; }

  /* ---------- top bar ---------- */
  .topbar {
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 14px;
    padding: 7px 12px;
  }
  .brand { display: flex; align-items: baseline; gap: 10px; }
  h1 { font-size: 16px; letter-spacing: -0.01em; white-space: nowrap; }
  h1 .dim { color: var(--faint); font-weight: 500; }
  .formula { font-size: 12px; white-space: nowrap; }
  .formula b { font-weight: 700; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .data-input { flex: 1; min-width: 120px; font-size: 12px; padding: 5px 8px; }
  .toggle-group button { padding: 4px 9px; font-size: 12px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; cursor: pointer; }
  .cb input { accent-color: var(--model); }
  .chars { font-size: 11px; white-space: nowrap; }

  /* ---------- body: fills the rest of the viewport ---------- */
  .body {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr);
    grid-template-rows: minmax(0, 1.15fr) minmax(0, 0.85fr);
    gap: 8px;
  }
  .stream-panel { grid-column: 1; grid-row: 1; padding: 10px 12px; }
  .cands-panel { grid-column: 1; grid-row: 2; padding: 10px 12px; }
  .side { grid-column: 2; grid-row: 1 / 3; display: flex; flex-direction: column; gap: 8px; min-height: 0; }
  .cost-panel { flex: 0 0 auto; padding: 10px 12px; }
  .chart-panel { flex: 1 1 auto; min-height: 0; padding: 10px 12px; }
  .row-title { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }

  /* ---------- transport ---------- */
  .transport-panel {
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

  @media (max-width: 880px) {
    .body { grid-template-columns: 1fr; grid-template-rows: 1fr 1fr 1fr; }
    .side { grid-column: 1; grid-row: 3; }
  }
</style>
