<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { trace } from '../lib/mdl/engine';
  import type { Step } from '../lib/mdl/types';
  import {
    graphProblem,
    parseEdgeList,
    defaultConfig,
    selfCheck,
    type GraphModel,
    type SubMove,
    type CodeMode
  } from '../lib/graph/graph';
  import { forceLayout, hashSeed } from '../lib/graph/layout';

  import GraphView from './graph/GraphView.svelte';
  import SubstructuresView from './graph/SubstructuresView.svelte';
  import CostPanel from './CostPanel.svelte';
  import CostChart from './CostChart.svelte';
  import CandidatesTable from './CandidatesTable.svelte';
  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';

  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('graph', [
    { id: 'graph', title: 'Graph' },
    { id: 'cands', title: 'Candidate substructures' },
    { id: 'subs', title: 'Dictionary' },
    { id: 'cost', title: 'Description length' },
    { id: 'chart', title: 'Evolution' },
    { id: 'input', title: 'Edge list (data)' }
  ]);

  // A directed 3-cycle of identically-labeled nodes — the repeated motif.
  const tri = (i: number) =>
    `t${i}a:o t${i}b:o e\nt${i}b:o t${i}c:o e\nt${i}c:o t${i}a:o e`;
  // A directed 4-cycle (square).
  const sq = (i: number) =>
    `s${i}1:q s${i}2:q g\ns${i}2:q s${i}3:q g\ns${i}3:q s${i}4:q g\ns${i}4:q s${i}1:q g`;
  // A 3-armed star: a hub with three leaves.
  const star = (i: number) =>
    `st${i}h:h st${i}1:p arm\nst${i}h:h st${i}2:p arm\nst${i}h:h st${i}3:p arm`;
  // n triangles linked head-to-tail — collapsing them reveals a higher-order
  // chain that itself compresses, and so on (a multi-level cascade).
  const chain = (n: number) => {
    const parts: string[] = [];
    for (let i = 0; i < n; i++) parts.push(tri(i));
    for (let i = 0; i < n - 1; i++) parts.push(`t${i}c:o t${i + 1}a:o link`);
    return parts.join('\n');
  };
  // A nuclear family: married M+F parents, each child parented by both. Node
  // labels are sex (M/F); ids are readable names. The kid SEX is part of the
  // label, so a two-son family and a two-daughter family are the SAME shape but
  // DIFFERENT substructures — the point of label-driven matching.
  const family = (sur: string, kids: ('M' | 'F')[]) => {
    const lines = [`# the ${sur} family`, `${sur}_dad:M ${sur}_mom:F married`];
    kids.forEach((sex, j) => {
      const kid = `${sur}_${sex === 'M' ? 'son' : 'dau'}${j + 1}:${sex}`;
      lines.push(`${sur}_dad:M ${kid} parent`);
      lines.push(`${sur}_mom:F ${kid} parent`);
    });
    return lines.join('\n');
  };

  // Dependency-grammar sample: node ids are words, LABELS are parts of speech
  // (D/A/N/V), edges are typed relations. Unlike the all-`o` triangle graphs, the
  // match is driven by labels + relations — so MDL discovers the NOUN PHRASE
  // (N←det D, N←amod A) first, then the CLAUSE (V with subject- and object-NPs)
  // on top of it: linguistic constituency falling out of compression.
  const dependency = `# the big dog chased the small cat
chased:V dog:N subj
chased:V cat:N obj
dog:N the1:D det
dog:N big:A amod
cat:N the2:D det
cat:N small:A amod
# a tall man read a long book
read:V man:N subj
read:V book:N obj
man:N a1:D det
man:N tall:A amod
book:N a2:D det
book:N long:A amod
# this old king saw that grand hall
saw:V king:N subj
saw:V hall:N obj
king:N this1:D det
king:N old:A amod
hall:N that1:D det
hall:N grand:A amod`;

  const SAMPLES: Record<string, string> = {
    // Gentle intro: one triangle collapses everywhere, then the collapsed nodes
    // form a pair that itself compresses.
    '4 triangles (chain)': chain(4),
    // Multi-level hierarchy: △×16 → chains of S0 → chains of S1 …
    'deep cascade (16 △)': chain(16),
    // Three DISTINCT substructures discovered in turn (square, star, triangle).
    'varied: △ ◻ ★ (×3)':
      [tri(0), tri(1), tri(2), sq(0), sq(1), sq(2), star(0), star(1), star(2)].join('\n'),
    // Labels carry meaning: the NP and the clause emerge as substructures.
    'dependency grammar (NP → clause)': dependency,
    // Same shape, different labels: two-son and two-daughter families are
    // detected as DISTINCT substructures because sex (M/F) is part of the match.
    'kinship (♂♂ vs ♀♀ families)': [
      family('Stone', ['M', 'M']),
      family('Reed', ['M', 'M']),
      family('Vale', ['M', 'M']),
      family('Cole', ['F', 'F']),
      family('Ash', ['F', 'F']),
      family('Bell', ['F', 'F'])
    ].join('\n'),
    // MDL ≠ frequency: the triangle wins over the more-frequent {q,q} edge.
    'two motifs: △×3 + ◻×2':
      [tri(0), tri(1), tri(2), sq(0), sq(1), 't0a:o s01:q bridge', 't2c:o s11:q bridge'].join('\n')
  };

  const DEFAULT_SAMPLE = '4 triangles (chain)';
  let sampleKey = $state(DEFAULT_SAMPLE);
  let text = $state(SAMPLES[DEFAULT_SAMPLE]);
  let codeMode = $state<CodeMode>('uniform');
  let includeOverhead = $state(true);

  const player = new Player<Step<GraphModel, SubMove>>();

  // The layout depends only on the base graph's shape, not on the code config,
  // so it's stable across uniform/Shannon/overhead toggles.
  const layout = $derived.by(() => {
    const base = parseEdgeList(text || ' ', defaultConfig);
    return forceLayout(
      base.nodes.length,
      base.edges.map((e) => [e.src, e.dst] as [number, number]),
      hashSeed(text)
    );
  });

  $effect(() => {
    const config = { ...defaultConfig, codeMode, includeOverhead };
    const problem = graphProblem(text || ' ', config);
    player.load(trace(problem, { maxSteps: 200 }));
  });

  if (import.meta.env.DEV) {
    const r = selfCheck();
    (r.ok ? console.log : console.warn)(r.msg);
  }

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
    <span class="lbl">graph</span>
    <select value={sampleKey} onchange={(e) => pickSample((e.currentTarget as HTMLSelectElement).value)}>
      {#each Object.keys(SAMPLES) as k}<option value={k}>{k}</option>{/each}
    </select>
  </label>

  <div class="f">
    <span class="lbl">code</span>
    <div class="toggle-group">
      <button class:active={codeMode === 'uniform'} onclick={() => (codeMode = 'uniform')}>log₂V</button>
      <button class:active={codeMode === 'shannon'} onclick={() => (codeMode = 'shannon')}>−log₂p</button>
    </div>
  </div>

  <label class="cb" title="count the bits to transmit the model-of-the-model (dictionary framing / code table)">
    <input type="checkbox" bind:checked={includeOverhead} /> overhead
  </label>

  {#if cur}
    <span class="chars mono muted">{cur.model.nodes.length}n · {cur.model.edges.length}e</span>
  {/if}

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if cur}
  <PanelHost manager={panels}>
    <div class="col col-dict">
      <Panel manager={panels} id="subs" weight={1}>
        {#snippet actions()}
          <span class="mono">{cur.model.subs.length} sub{cur.model.subs.length === 1 ? '' : 's'}</span>
        {/snippet}
        <SubstructuresView model={cur.model} />
      </Panel>
    </div>

    <div class="col col-mid">
      <Panel manager={panels} id="graph" weight={1.5}>
        {#snippet actions()}
          <span class="mono">{cur.model.subs.length} sub{cur.model.subs.length === 1 ? '' : 's'}</span>
        {/snippet}
        <GraphView model={cur.model} {layout} chosen={cur.chosen} />
      </Panel>

      <Panel manager={panels} id="cands" weight={0.8}>
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

      <Panel manager={panels} id="chart" weight={0.9}>
        <CostChart steps={player.steps} index={player.index} onSeek={(i) => player.seek(i)} />
      </Panel>

      <Panel manager={panels} id="input" weight={0.8}>
        <textarea
          class="edge-input mono scrollbar"
          bind:value={text}
          spellcheck="false"
          placeholder={'src:Label  dst:Label  EDGELABEL\none edge per line'}
        ></textarea>
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
  .toggle-group button { padding: 4px 9px; font-size: 12px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; cursor: pointer; }
  .cb input { accent-color: var(--model); }
  .chars { font-size: 11px; white-space: nowrap; }
  .reset-layout { padding: 4px 9px; font-size: 11px; white-space: nowrap; }

  .col { display: flex; flex-direction: column; gap: 8px; min-height: 0; min-width: 0; }
  .col-dict { flex: 0.5 1 0; }
  .col-mid { flex: 1.5 1 0; }
  .col-right { flex: 1 1 0; }

  .edge-input {
    flex: 1 1 auto; min-height: 0; width: 100%; resize: none;
    background: var(--bg-2); color: var(--text);
    border: 1px solid var(--border); border-radius: var(--r-sm);
    padding: 8px; font-size: 12px; line-height: 1.5;
  }

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
