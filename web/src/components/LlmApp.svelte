<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import { trainTrace, type LlmStep } from '../lib/llm/trainTrace';
  import { DATASETS, DATASET_KEYS } from '../lib/llm/datasets';
  import { tokenColor } from '../lib/llm/colors';
  import { makeLab, type Lab, type EvalCase } from '../lib/llm/lab';
  import type { FactSpec } from '../lib/llm/implant';
  import { router } from '../lib/router.svelte';

  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import ActGrid from './llm/ActGrid.svelte';
  import AttentionView from './llm/AttentionView.svelte';
  import OutputBars from './llm/OutputBars.svelte';
  import LossChart from './llm/LossChart.svelte';
  import LensView from './llm/LensView.svelte';
  import ImplantPanel from './llm/ImplantPanel.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import MathLink from './math/MathLink.svelte';

  const panels = new PanelManager(
    'llm',
    [
      { id: 'tokens', title: 'Tokens', fit: true },
      { id: 'embed', title: 'Token embeddings' },
      { id: 'pos', title: 'With position' },
      { id: 'attn', title: 'Attention' },
      { id: 'ffn', title: 'Feed-forward' },
      { id: 'implant', title: 'Implant facts' },
      { id: 'output', title: 'Next-token guess' },
      { id: 'lens', title: 'Logit lens' },
      { id: 'loss', title: 'Loss over training' },
      { id: 'explain', title: 'Readout', fit: true },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [
        ['tokens', 'embed', 'pos', 'loss'],
        ['attn', 'ffn', 'implant'],
        ['output', 'lens', 'explain', 'guide']
      ],
      widths: [1, 1.3, 1.1],
      weights: { loss: 0.8, attn: 1.5, implant: 1.1, output: 1.3, lens: 1.2 }
    }
  );

  const STEPS = 300;
  const dsParam = router.get('ds');
  const initialDs = dsParam && DATASETS[dsParam] ? dsParam : DATASET_KEYS[0];
  const initialQ = router.get('q');
  let datasetKey = $state(initialDs);
  let seed = $state(router.num('seed') ?? 1);

  const ds = $derived(DATASETS[datasetKey]);
  const player = new Player<LlmStep>();

  // The fixed-probe trace + the lab (replay + implant surgery over the per-step
  // weights), both rebuilt only when the dataset or seed changes — never on
  // playback.
  let lab = $state<Lab | null>(null);

  $effect(() => {
    const d = DATASETS[datasetKey];
    const run = trainTrace(d, { steps: STEPS, seed });
    player.load(run.steps);
    const toIds = (words: string[]) => words.map((w) => d.vocab.indexOf(w));
    lab = makeLab(
      run,
      d.trainData.map((seq) => toIds(seq.slice(0, -1)))
    );
  });

  // --- Query box: replay an arbitrary prompt against the current step's model.
  let queryText = $state(initialQ ?? DATASETS[initialDs].probe.join(' '));
  let committedQuery = $state<number[] | null>(null);

  // --- Implanted facts: applied on top of whichever step is selected.
  let facts = $state<FactSpec[]>([]);
  let factSeq = 0;

  // Reset the query and implants whenever the dataset (and vocabulary) CHANGES —
  // guarded so the first run doesn't clobber a URL-restored query.
  let prevDs = initialDs;
  $effect(() => {
    if (datasetKey === prevDs) return;
    prevDs = datasetKey;
    queryText = DATASETS[datasetKey].probe.join(' ');
    committedQuery = null;
    facts = [];
  });

  $effect(() => {
    router.setQuery({
      ds: datasetKey === DATASET_KEYS[0] ? null : datasetKey,
      seed: seed === 1 ? null : seed,
      q: committedQuery !== null ? queryText : null
    });
  });

  function parseQuery(text: string): { ids?: number[]; error?: string } {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length === 0) return { error: 'type a prompt' };
    if (words.length > ds.ctxLen) return { error: `too long — max ${ds.ctxLen} tokens` };
    const ids: number[] = [];
    for (const w of words) {
      const id = ds.vocab.indexOf(w);
      if (id === -1) return { error: `unknown token “${w}”` };
      ids.push(id);
    }
    return { ids };
  }

  const parsed = $derived(parseQuery(queryText));
  const parseError = $derived(queryText.trim() === '' ? '' : (parsed.error ?? ''));

  // Restore a committed query from the URL (only if it still parses).
  if (initialQ) {
    const p = parseQuery(initialQ);
    if (p.ids) committedQuery = p.ids;
  }

  function runQuery() {
    if (parsed.ids) committedQuery = parsed.ids;
  }
  function clearQuery() {
    committedQuery = null;
    queryText = ds.probe.join(' ');
  }

  const cur = $derived(player.current);
  const isQuery = $derived(committedQuery !== null);

  const idsOf = (words: string[]) => words.map((w) => ds.vocab.indexOf(w));

  // The input currently on display: the user's committed query, else the probe.
  const activeIds = $derived(committedQuery ?? idsOf(ds.probe));

  // Forward pass through the current step's model — with implants grafted in.
  const analysis = $derived.by(() => (lab ? lab.forward(player.index, activeIds, facts) : null));
  const activeViz = $derived(analysis?.viz ?? null);

  // Lazily computed (only while its panel is open): the rung-by-rung readout.
  const lensReport = $derived.by(() => (lab && analysis ? lab.lens(player.index, activeIds, facts) : null));

  // Description length of every training sequence + probe, before vs after the
  // implants. Whole sequences are charged at every transition — L(seq|M) — so
  // damage anywhere in a sequence shows up, not just at its final token.
  const interferenceRows = $derived.by(() => {
    if (!lab || facts.length === 0) return [];
    const cases: EvalCase[] = [
      ...facts.map((f) => ({
        label: `${f.promptIds.map((id) => ds.vocab[id]).join(' ')} → ${ds.vocab[f.targetId]}`,
        inputIds: f.promptIds,
        targets: [{ pos: f.promptIds.length - 1, id: f.targetId }],
        isFact: true
      })),
      ...ds.trainData.map((seq) => {
        const t = idsOf(seq);
        return {
          label: seq.join(' '),
          inputIds: t.slice(0, -1),
          targets: t.slice(1).map((id, pos) => ({ pos, id }))
        };
      }),
      {
        label: `${ds.probe.join(' ')} → ${ds.probeTarget} ⟨probe⟩`,
        inputIds: idsOf(ds.probe),
        targets: [{ pos: ds.probe.length - 1, id: ds.vocab.indexOf(ds.probeTarget) }]
      }
    ];
    return lab.interference(player.index, facts, cases);
  });

  function addFact(f: Omit<FactSpec, 'key'>) {
    facts = [...facts, { ...f, key: `f${factSeq++}` }];
  }
  function removeFact(key: string) {
    facts = facts.filter((f) => f.key !== key);
  }

  const activeTokens = $derived(activeViz ? activeViz.tokenIds.map((id) => ds.vocab[id]) : []);
  const activeColors = $derived(activeTokens.map((t) => tokenColor(ds, t)));
  const vocabColors = $derived(ds.vocab.map((t) => tokenColor(ds, t)));

  const activePred = $derived.by(() => {
    if (!activeViz) return { id: 0, token: '', conf: 0 };
    let id = 0;
    for (let v = 1; v < activeViz.probs.length; v++) if (activeViz.probs[v] > activeViz.probs[id]) id = v;
    return { id, token: ds.vocab[id], conf: activeViz.probs[id] };
  });

  const stepLoss = $derived(cur?.loss ?? 0);
  const note = $derived(cur?.note ?? '');
  // Read these off the ACTIVE forward pass (not the precomputed trace) so they
  // stay truthful when implants are grafted in.
  const probeLearned = $derived(activePred.token === ds.probeTarget);
  const activeTargetProb = $derived(activeViz ? activeViz.probs[ds.vocab.indexOf(ds.probeTarget)] : 0);
</script>

<TopBar {panels}>
  <label class="f">
    <span class="lbl">dataset</span>
    <select value={datasetKey} onchange={(e) => (datasetKey = (e.currentTarget as HTMLSelectElement).value)}>
      {#each DATASET_KEYS as k}<option value={k}>{DATASETS[k].label}</option>{/each}
    </select>
  </label>

  <div class="f query">
    <span class="lbl">prompt</span>
    <input
      class="q-input mono"
      class:invalid={!!parseError}
      bind:value={queryText}
      onkeydown={(e) => e.key === 'Enter' && runQuery()}
      spellcheck="false"
      placeholder={ds.probe.join(' ')}
      title={`vocabulary: ${ds.vocab.join(' ')}`}
    />
    <button class="ghost" onclick={runQuery} disabled={!parsed.ids}>▶ run</button>
    {#if isQuery}
      <button class="ghost" onclick={clearQuery} title="Back to the training probe">✕ probe</button>
    {/if}
  </div>

  {#if parseError}
    <span class="qerr mono" title={`vocabulary: ${ds.vocab.join(' ')}`}>{parseError}</span>
  {:else}
    <span class="viewing mono" class:q={isQuery}>{isQuery ? 'your prompt' : 'training probe'}</span>
  {/if}

  <button class="ghost" title="Re-initialise weights with a new random seed" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  <span class="endspacer"></span>
  <MathLink lesson="cross-entropy" label="derive the loss" />
</TopBar>

{#if activeViz}
  {#snippet aTokens()}<span class="mono">{activeTokens.length} tokens · {isQuery ? 'query' : 'probe'}</span>{/snippet}
  {#snippet pTokens()}
    <div class="tokens">
      {#each activeTokens as t, i}
        <span class="tchip mono" style="color:{activeColors[i]}; border-color:{activeColors[i]}">
          <span class="tpos faint">{i}</span>{t}
        </span>
      {/each}
    </div>
  {/snippet}

  {#snippet aEmbed()}<span class="mono">{ds.embDim}d</span>{/snippet}
  {#snippet pEmbed()}
    <ActGrid matrix={activeViz!.tokEmb} rowLabels={activeTokens} rowColors={activeColors} signed />
  {/snippet}

  {#snippet aPos()}<span class="mono">token + position</span>{/snippet}
  {#snippet pPos()}
    <ActGrid matrix={activeViz!.embedded} rowLabels={activeTokens} rowColors={activeColors} signed />
  {/snippet}

  {#snippet aLoss()}<span class="mono">loss {stepLoss.toFixed(3)}</span>{/snippet}
  {#snippet pLoss()}
    <LossChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
  {/snippet}

  {#snippet aAttn()}<span class="mono">{ds.nHeads} heads</span>{/snippet}
  {#snippet pAttn()}
    <AttentionView attn={activeViz!.attn} tokens={activeTokens} tokenColors={activeColors} />
  {/snippet}

  {#snippet aFfn()}<span class="mono">{ds.ffnHid}{facts.length ? `+${facts.length}` : ''} units</span>{/snippet}
  {#snippet pFfn()}
    <ActGrid
      matrix={activeViz!.ffnHidden}
      rowLabels={activeTokens}
      rowColors={activeColors}
      signed={false}
      colLabel={facts.length ? `units (last ${facts.length} implanted)` : 'units'}
    />
  {/snippet}

  {#snippet aImplant()}<span class="mono">{facts.length} fact{facts.length === 1 ? '' : 's'}</span>{/snippet}
  {#snippet pImplant()}
    <ImplantPanel {ds} {facts} rows={interferenceRows} onAdd={addFact} onRemove={removeFact} />
  {/snippet}

  {#snippet aOutput()}<span class="mono">{(activePred.conf * 100).toFixed(0)}% peak</span>{/snippet}
  {#snippet pOutput()}
    <OutputBars probs={activeViz!.probs} vocab={ds.vocab} colors={vocabColors} predId={activePred.id} targetToken={isQuery ? '' : ds.probeTarget} />
  {/snippet}

  {#snippet aLens()}<span class="mono">−log₂ p by depth</span>{/snippet}
  {#snippet pLens()}
    {#if lensReport}
      <LensView report={lensReport} vocab={ds.vocab} colors={vocabColors} focusId={activePred.id} />
    {/if}
  {/snippet}

  {#snippet pExplain()}
    <div class="readout">
      {#if isQuery}
        <p class="say">
          Your prompt <span class="mono">“{activeTokens.join(' ')}”</span> at step <b>{player.index}</b>
          → the model predicts
          <b class="mono" style="color:{vocabColors[activePred.id]}">{activePred.token}</b>
          at <b>{(activePred.conf * 100).toFixed(0)}%</b>.
        </p>
        <p class="say muted">Scrub the timeline to watch this prediction form as the model trains.</p>
      {:else}
        <p class="say">
          After <b>{player.index}</b> step{player.index === 1 ? '' : 's'}, the model reads
          <span class="mono">“{ds.probe.join(' ')}”</span> and predicts
          <b class="mono" style="color:{vocabColors[activePred.id]}">{activePred.token}</b>
          at <b>{(activePred.conf * 100).toFixed(0)}%</b>.
        </p>
        <p class="say muted">
          Correct continuation <span class="mono" style="color:{tokenColor(ds, ds.probeTarget)}">{ds.probeTarget}</span>
          sits at <b>{(activeTargetProb * 100).toFixed(0)}%</b>
          {#if probeLearned}<span class="good"> — learned ✓</span>{/if}
        </p>
      {/if}
      {#if facts.length}
        <p class="say muted">
          <b>{facts.length}</b> implanted fact{facts.length === 1 ? '' : 's'} active — written
          straight into the FFN as key→value memory slots, no training. They re-apply at
          every step, so scrubbing shows the same surgery against different weights.
        </p>
      {/if}
    </div>
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="llm" sections={['minigpt']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{
      tokens: pTokens,
      embed: pEmbed,
      pos: pPos,
      loss: pLoss,
      attn: pAttn,
      ffn: pFfn,
      implant: pImplant,
      output: pOutput,
      lens: pLens,
      explain: pExplain,
      guide: pGuide
    }}
    actions={{
      tokens: aTokens,
      embed: aEmbed,
      pos: aPos,
      loss: aLoss,
      attn: aAttn,
      ffn: aFfn,
      implant: aImplant,
      output: aOutput,
      lens: aLens
    }}
  />

  <TransportBar {player} {note} converged={!cur?.chosen} />
{/if}

<style>
  .query { flex: 1; min-width: 220px; }
  .q-input { flex: 1; min-width: 120px; font-size: 12px; padding: 5px 8px; }
  .q-input.invalid { border-color: var(--bad); }
  .qerr { font-size: 11px; color: var(--bad); white-space: nowrap; }
  .viewing { font-size: 11px; color: var(--muted); white-space: nowrap; }
  .viewing.q { color: var(--chosen); }
  .endspacer { flex: 0 1 auto; margin-left: auto; }

  .tokens { display: flex; flex-wrap: wrap; gap: 5px; }
  .tchip {
    display: inline-flex; align-items: baseline; gap: 4px;
    font-size: 13px; padding: 3px 7px;
    border: 1px solid; border-radius: 5px; background: var(--bg-2);
  }
  .tpos { font-size: 9px; }

  .readout { display: flex; flex-direction: column; gap: 6px; }
  .say { margin: 0; font-size: 12.5px; line-height: 1.45; }
</style>
