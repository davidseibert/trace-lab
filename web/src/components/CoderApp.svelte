<script lang="ts">
  import { untrack } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import {
    uniformStream,
    empiricalStream,
    grammarStream,
    llmStream,
    clampStream,
    MAX_CHARS,
    type CodeStream
  } from '../lib/coder/coder';
  import { encode, decode, idealBits, type CoderStep } from '../lib/coder/arithmetic';
  import { grammarProblem, defaultConfig, type GrammarModel } from '../lib/string/grammar';
  import { trace } from '../lib/mdl/engine';
  import { trainTrace, makeForward } from '../lib/llm/trainTrace';
  import { DATASETS, DATASET_KEYS } from '../lib/llm/datasets';
  import { router } from '../lib/router.svelte';

  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import IntervalView from './coder/IntervalView.svelte';
  import DistView from './coder/DistView.svelte';
  import BitstreamView from './coder/BitstreamView.svelte';
  import BitsChart from './coder/BitsChart.svelte';

  const panels = new PanelManager(
    'coder',
    [
      { id: 'interval', title: 'Interval' },
      { id: 'chart', title: 'Bits accumulating' },
      { id: 'dist', title: 'Distribution' },
      { id: 'bits', title: 'Bitstream', zoomable: true },
      { id: 'explain', title: 'Readout', fit: true },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['interval', 'chart'], ['dist', 'bits', 'explain', 'guide']],
      widths: [1.3, 1],
      weights: { interval: 1.4 }
    }
  );

  type Source = 'builtin' | 'grammar' | 'llm';
  type Model = 'uniform' | 'empirical';
  type Mode = 'encode' | 'decode';

  const STEPS = 300;

  const srcParam = router.get('src');
  const modelParam = router.get('model');
  let raw = $state(router.get('s') ?? 'abracadabra');
  let source = $state<Source>(
    srcParam === 'grammar' || srcParam === 'llm' ? srcParam : 'builtin'
  );
  let model = $state<Model>(modelParam === 'uniform' ? 'uniform' : 'empirical');
  let mode = $state<Mode>(router.get('phase') === 'decode' ? 'decode' : 'encode');

  // Mini-GPT source controls.
  const dsParam = router.get('ds');
  let llmKey = $state(dsParam && DATASETS[dsParam] ? dsParam : DATASET_KEYS[0]);
  let llmSeqIdx = $state(router.num('seq') ?? 0);
  let llmStep = $state(router.num('tstep') ?? STEPS - 1); // default: fully trained

  $effect(() => {
    router.setQuery({
      src: source === 'builtin' ? null : source,
      s: source === 'llm' || raw === 'abracadabra' ? null : raw,
      model: source !== 'builtin' || model === 'empirical' ? null : model,
      phase: mode === 'encode' ? null : mode,
      ds: source === 'llm' && llmKey !== DATASET_KEYS[0] ? llmKey : null,
      seq: source === 'llm' && llmSeqIdx !== 0 ? llmSeqIdx : null,
      tstep: source === 'llm' && llmStepClamped !== STEPS - 1 ? llmStepClamped : null
    });
  });

  const player = new Player<CoderStep>();

  // Keep within float-precision limits; surface if we trimmed the input.
  const chars = $derived([...raw]);
  const text = $derived(chars.slice(0, MAX_CHARS).join(''));
  const tooLong = $derived(chars.length > MAX_CHARS);

  // Grammar source: compress the text to convergence (Shannon), then code the
  // resulting symbol sequence.
  const grammarModel = $derived.by<GrammarModel | null>(() => {
    if (source !== 'grammar' || text.length === 0) return null;
    const steps = trace(grammarProblem(text, { ...defaultConfig, codeMode: 'shannon' }));
    return steps[steps.length - 1]?.model ?? null;
  });

  // Mini-GPT source. The training run depends only on the dataset, so scrubbing
  // the training-step slider doesn't retrain — it just re-runs one forward pass.
  const llmDs = $derived(DATASETS[llmKey]);
  const llmRun = $derived.by(() =>
    source === 'llm' ? trainTrace(llmDs, { steps: STEPS, seed: 1 }) : null
  );
  const llmForward = $derived(llmRun ? makeForward(llmRun) : null);
  const llmSeq = $derived(llmDs.trainData[Math.min(llmSeqIdx, llmDs.trainData.length - 1)] ?? []);
  const llmTokenIds = $derived(llmSeq.map((w) => llmDs.vocab.indexOf(w)));
  const llmStepClamped = $derived(Math.max(0, Math.min(STEPS - 1, llmStep)));
  const llmViz = $derived(llmForward ? llmForward(llmStepClamped, llmTokenIds) : null);

  const rawStream = $derived.by<CodeStream>(() => {
    if (source === 'grammar') return grammarModel ? grammarStream(grammarModel) : [];
    if (source === 'llm') return llmViz ? llmStream(llmViz, llmDs.vocab) : [];
    return model === 'uniform' ? uniformStream(text) : empiricalStream(text);
  });

  const clamped = $derived(clampStream(rawStream));
  const enc = $derived(encode(clamped.stream));
  const dec = $derived(decode(enc.codeword.value, clamped.stream));
  const activeSteps = $derived(mode === 'encode' ? enc.steps : dec);
  const idealTotal = $derived(idealBits(enc.steps));
  const roundTripOk = $derived(dec.every((s) => !s.mismatch));

  // Swap in the recomputed trace but KEEP the current coder position (clamped),
  // so changing the training step (or model/phase) holds the step you're on
  // instead of snapping back to 0. The two timelines stay independent.
  $effect(() => {
    const steps = activeSteps;
    untrack(() => {
      player.steps = steps;
      player.seek(player.index);
    });
  });

  const cur = $derived(player.current);
  const atEnd = $derived(player.atEnd);
  const note = $derived(cur?.note ?? '');

  // Grammar compression summary, when that source is active.
  const gramInfo = $derived.by(() => {
    if (source !== 'grammar' || !grammarModel) return '';
    return `compressed ${text.length} chars → ${grammarModel.sequence.length} symbols, ${grammarModel.rules.length} rules`;
  });

  const llmInfo = $derived.by(() => {
    if (source !== 'llm' || !llmRun) return '';
    const loss = llmRun.steps[llmStepClamped]?.loss ?? 0;
    return `step ${llmStepClamped}/${STEPS - 1} · loss ${loss.toFixed(3)}`;
  });

  // What's being coded, for the readout.
  const subject = $derived(source === 'llm' ? llmSeq.join(' ') : text);
  const memoryless = $derived(source !== 'llm');
</script>

<TopBar {panels}>
  <div class="f">
    <span class="lbl">source</span>
    <div class="toggle-group">
      <button class:active={source === 'builtin'} onclick={() => (source = 'builtin')}>Built-in</button>
      <button class:active={source === 'grammar'} onclick={() => (source = 'grammar')}>Grammar</button>
      <button class:active={source === 'llm'} onclick={() => (source = 'llm')}>Mini·GPT</button>
    </div>
  </div>

  {#if source === 'llm'}
    <label class="f">
      <span class="lbl">dataset</span>
      <select value={llmKey} onchange={(e) => { llmKey = (e.currentTarget as HTMLSelectElement).value; llmSeqIdx = 0; }}>
        {#each DATASET_KEYS as k}<option value={k}>{DATASETS[k].label}</option>{/each}
      </select>
    </label>
    <label class="f">
      <span class="lbl">sequence</span>
      <select value={llmSeqIdx} onchange={(e) => (llmSeqIdx = +(e.currentTarget as HTMLSelectElement).value)}>
        {#each llmDs.trainData as seq, i (i)}<option value={i}>{seq.join(' ')}</option>{/each}
      </select>
    </label>
    <label class="f train">
      <span class="lbl">train step</span>
      <input type="range" min="0" max={STEPS - 1} value={llmStepClamped} oninput={(e) => (llmStep = +(e.currentTarget as HTMLInputElement).value)} />
      <span class="hint mono">{llmInfo}</span>
    </label>
  {:else}
    <label class="f">
      <span class="lbl">string</span>
      <input class="s-input mono" bind:value={raw} spellcheck="false" placeholder="type a short string" />
    </label>
    {#if source === 'builtin'}
      <div class="f">
        <span class="lbl">model</span>
        <div class="toggle-group">
          <button class:active={model === 'uniform'} onclick={() => (model = 'uniform')}>Uniform</button>
          <button class:active={model === 'empirical'} onclick={() => (model = 'empirical')}>Empirical</button>
        </div>
      </div>
    {:else}
      <span class="hint mono">{gramInfo}</span>
    {/if}
  {/if}

  <div class="f">
    <span class="lbl">phase</span>
    <div class="toggle-group">
      <button class:active={mode === 'encode'} onclick={() => (mode = 'encode')}>Encode</button>
      <button class:active={mode === 'decode'} onclick={() => (mode = 'decode')}>Decode</button>
    </div>
  </div>

  <span class="spacer"></span>

  {#if tooLong}
    <span class="warn mono" title="float intervals stay exact only ~52 bits deep">first {MAX_CHARS} chars</span>
  {/if}
  {#if clamped.truncated}
    <span class="warn mono" title="trimmed so the interval stays above float resolution">precision-trimmed</span>
  {/if}
  {#if !roundTripOk}
    <span class="warn bad mono">round-trip mismatch</span>
  {/if}
</TopBar>

{#if cur}
  {#snippet aInterval()}<span class="mono">{mode} · step {player.index + 1}/{player.count}</span>{/snippet}
  {#snippet pInterval()}
    <IntervalView steps={player.steps} index={player.index} {memoryless} />
  {/snippet}

  {#snippet aChart()}<span class="mono">{idealTotal.toFixed(1)} ideal · {enc.codeword.nbits} bits</span>{/snippet}
  {#snippet pChart()}
    <BitsChart steps={player.steps} index={player.index} refBits={enc.codeword.nbits} onSeek={(i) => player.seek(i)} />
  {/snippet}

  {#snippet aDist()}<span class="mono">{cur!.dist.length} symbols</span>{/snippet}
  {#snippet pDist()}
    <DistView step={cur!} />
  {/snippet}

  {#snippet aBits()}<span class="mono">{cur!.bitsSoFar.toFixed(2)} bits</span>{/snippet}
  {#snippet pBits()}
    <BitstreamView step={cur!} codeword={enc.codeword} {idealTotal} final={atEnd} />
  {/snippet}

  {#snippet pExplain()}
    <div class="readout">
          {#if mode === 'encode'}
            <p class="say">
              Encoding <span class="mono">“{subject}”</span>: each symbol narrows the
              interval by its probability, adding <b>−log₂p</b> bits.
              {#if atEnd}
                Done — the message lives in
                <span class="mono">[{cur!.newLo.toFixed(5)}, {cur!.newHi.toFixed(5)})</span>,
                named by <b class="mono" style="color:var(--total)">0.{enc.codeword.bits}</b>
                (<b>{enc.codeword.nbits}</b> bits vs. ideal <b>{idealTotal.toFixed(2)}</b>).
              {:else}
                So far <b>{cur!.bitsSoFar.toFixed(2)}</b> bits.
              {/if}
            </p>
          {:else}
            <p class="say">
              Decoding the single number <b class="mono" style="color:var(--chosen)">{enc.codeword.value.toFixed(5)}</b>:
              at each step we find which slice it lands in, emit that symbol, and zoom in.
              Recovered so far: <span class="mono">“{cur!.emitted.join('')}”</span>
              {#if atEnd}<span class="good"> — full round-trip ✓</span>{/if}
            </p>
          {/if}
          <p class="say muted">
            The decoder needs the <i>same</i> distribution the encoder used — which is
            exactly why MDL charges L(M): the model has to be transmitted too.
          </p>
        </div>
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="coder" sections={['coderread']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ interval: pInterval, chart: pChart, dist: pDist, bits: pBits, explain: pExplain, guide: pGuide }}
    actions={{ interval: aInterval, chart: aChart, dist: aDist, bits: aBits }}
  />

  <TransportBar {player} {note} converged={atEnd} />
{:else}
  <div class="panel empty">
    <p class="muted">Type a short string to encode.</p>
  </div>
{/if}

<style>
  .s-input { width: 160px; font-size: 12px; padding: 5px 8px; }
  .train input[type='range'] { width: 120px; accent-color: var(--model); }

  .readout { display: flex; flex-direction: column; gap: 6px; }
  .say { margin: 0; font-size: 12.5px; line-height: 1.45; }

  .empty { flex: 1 1 auto; display: grid; place-items: center; }
</style>
