<script lang="ts">
  import { Player } from '../../lib/player.svelte';
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';
  import { chartScale } from '../../lib/chart';
  import { engine } from '../../lib/logit/engine.svelte';
  import { fetchHopfield, type HopfieldHeadsResponse, type HopfieldRegime } from '../../lib/logit/api';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import EngineOffline from '../shell/EngineOffline.svelte';
  import EngineStatus from '../shell/EngineStatus.svelte';

  const panels = new PanelManager(
    'hopfieldreal',
    [
      { id: 'grid', title: 'Regime map — layer × head' },
      { id: 'census', title: 'Regime census by layer' },
      { id: 'inspect', title: 'Head inspector' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['grid'], ['census', 'inspect', 'guide']],
      widths: [1.5, 1],
      weights: { grid: 2, census: 1.3, inspect: 1.3 }
    }
  );

  // Repetition on purpose: copy-pattern prompts give the retrieval story
  // something to retrieve (induction heads light up as sharp Hopfield reads).
  const DEFAULT_PROMPT =
    'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is';
  const DEFAULT_MODEL = 'gpt2';

  let modelName = $state(router.get('m') ?? DEFAULT_MODEL);
  let prompt = $state(router.get('t') ?? DEFAULT_PROMPT);
  let pos = $state(router.num('pos') ?? -1);

  $effect(() => {
    router.setQuery({
      m: modelName === DEFAULT_MODEL ? null : modelName,
      t: prompt === DEFAULT_PROMPT ? null : prompt,
      pos: pos === -1 ? null : pos
    });
  });

  let resp = $state<HopfieldHeadsResponse | null>(null);
  let running = $state(false);
  let err = $state('');
  let selected = $state<{ layer: number; head: number } | null>(null);

  // The Player scrubs the γ index — turning the temperature knob on a real
  // model. Starts at γ = 1 (the model's own 1/√d_k), not at the sweep edge.
  const player = new Player<number>();

  async function run() {
    if (running) return;
    running = true;
    err = '';
    try {
      const r = await fetchHopfield({ model: modelName, prompt, pos });
      engine.markUp();
      resp = r;
      selected = null;
      player.load(r.gammas);
      const gi = r.gammas.indexOf(1);
      if (gi >= 0) player.seek(gi);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
    running = false;
  }

  const gIdx = $derived(player.index);
  const gamma = $derived(resp?.gammas[gIdx] ?? 1);

  // heads[] is layer-major (the engine loops layers, then heads).
  const headAt = (L: number, h: number) => resp!.heads[L * resp!.n_heads + h];

  const REGIME_VAR: Record<HopfieldRegime, string> = {
    retrieval: 'var(--good)',
    metastable: 'var(--data)',
    global: 'var(--bad)'
  };

  const census = $derived.by(() => {
    if (!resp) return [];
    return [...Array(resp.n_layers).keys()].map((L) => {
      const c = { retrieval: 0, metastable: 0, global: 0 };
      for (let h = 0; h < resp!.n_heads; h++) c[headAt(L, h).curves[gIdx].regime]++;
      return c;
    });
  });

  const totals = $derived.by(() => {
    const t = { retrieval: 0, metastable: 0, global: 0 };
    for (const c of census) {
      t.retrieval += c.retrieval;
      t.metastable += c.metastable;
      t.global += c.global;
    }
    return t;
  });

  const selHead = $derived(selected && resp ? headAt(selected.layer, selected.head) : null);

  // Inspector chart: the selected head's entropy (normalized by log₂ seq) and
  // max weight across the γ sweep. Click seeks the player's γ index.
  const CW = 520;
  const CH = 150;
  const iScale = $derived(chartScale({ n: resp?.gammas.length ?? 1, max: 1, W: CW, H: CH }));
  const iEnt = $derived(
    selHead && resp ? selHead.curves.map((c) => c.entropy_bits / Math.max(1e-9, Math.log2(resp!.seq))) : []
  );
  const iMax = $derived(selHead ? selHead.curves.map((c) => c.max_w) : []);
</script>

<TopBar {panels}>
  <label class="f">
    <span class="lbl">model</span>
    <select value={modelName} onchange={(e) => (modelName = (e.currentTarget as HTMLSelectElement).value)}>
      <optgroup label="hub">
        {#each engine.hub as m (m.name)}<option value={m.name}>{m.name}</option>{/each}
      </optgroup>
      {#if !engine.models.some((m) => m.name === modelName)}
        <option value={modelName}>{modelName}</option>
      {/if}
    </select>
  </label>

  <label class="f prompt">
    <span class="lbl">prompt</span>
    <input class="p-input mono" bind:value={prompt} spellcheck="false" onkeydown={(e) => e.key === 'Enter' && run()} />
  </label>

  <label class="f">
    <span class="lbl" title="Destination position whose attention rows to read; -1 = the last token">pos</span>
    <input class="pos-input mono" type="number" min="-1" bind:value={pos} />
  </label>

  <button onclick={run} disabled={running}>{running ? 'reading…' : '▶ read heads'}</button>

  <!-- On entry: if the engine is up, run the restored/default prompt so the
       lens is alive (deterministic — a URL-restored run reproduces itself). -->
  <EngineStatus onProbe={(ok) => { if (ok && !resp) void run(); }} />
</TopBar>

{#if !engine.up && !resp}
  <EngineOffline
    what="This lens reads every attention head of a real model as one-step Hopfield retrieval."
    onUp={() => void run()}
  />
{:else}
  {#snippet aGrid()}
    {#if resp}<span class="mono">dst “{resp.tokens[resp.tokens.length - 1]}” · seq {resp.seq} · γ = {gamma}</span>{/if}
  {/snippet}
  {#snippet pGrid()}
    {#if resp}
      <div class="gwrap scrollbar">
        <div class="legend mono faint">
          <span><i style="background:var(--good)"></i>retrieval {totals.retrieval}</span>
          <span><i style="background:var(--data)"></i>metastable {totals.metastable}</span>
          <span><i style="background:var(--bad)"></i>global {totals.global}</span>
        </div>
        {#each [...Array(resp.n_layers).keys()] as L (L)}
          <div class="grow">
            <span class="glab mono faint">L{L}</span>
            <div class="gcells" style="grid-template-columns:repeat({resp.n_heads}, minmax(0, 26px))">
              {#each [...Array(resp.n_heads).keys()] as h (h)}
                {@const c = headAt(L, h).curves[gIdx]}
                <button
                  class="gcell"
                  class:sel={selected?.layer === L && selected?.head === h}
                  style="background:{REGIME_VAR[c.regime]}; opacity:{0.35 + 0.65 * c.max_w}"
                  title={`L${L} H${h} · H ${c.entropy_bits} bits · eff ${c.eff_k} · max ${c.max_w} · ${c.regime}`}
                  onclick={() => (selected = { layer: L, head: h })}
                  aria-label={`inspect layer ${L} head ${h}`}
                ></button>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    {:else if err}
      <div class="panel empty"><p class="mono">{err}</p><button class="ghost" onclick={run}>retry</button></div>
    {:else}
      <div class="panel empty">
        <p class="faint">Run a prompt to read every head’s attention row as a Hopfield retrieval step.</p>
        <button class="ghost" onclick={run} disabled={running}>{running ? 'reading…' : '▶ read heads'}</button>
      </div>
    {/if}
  {/snippet}

  {#snippet pCensus()}
    {#if resp}
      <div class="pcol scrollbar">
        {#each census as c, L (L)}
          <div class="crow">
            <span class="glab mono faint">L{L}</span>
            <div class="cbar">
              <div style="flex:{c.retrieval}; background:var(--good)"></div>
              <div style="flex:{c.metastable}; background:var(--data)"></div>
              <div style="flex:{c.global}; background:var(--bad)"></div>
            </div>
          </div>
        {/each}
        <div class="faint hint">Fraction of heads per regime at γ = {gamma}. The paper’s observation, in the wild: middle layers pool (metastable); sharp retrieval heads cluster where copying/induction lives.</div>
      </div>
    {:else}
      <div class="panel empty"><span class="faint">no run yet</span></div>
    {/if}
  {/snippet}

  {#snippet aInspect()}{#if selected}<span class="mono">L{selected.layer} H{selected.head}</span>{/if}{/snippet}
  {#snippet pInspect()}
    {#if selHead && resp}
      <div class="pcol">
        <div class="legend mono faint">
          <span><i style="background:var(--model)"></i>max weight</span>
          <span><i style="background:var(--data)"></i>entropy / log₂seq</span>
        </div>
        <svg viewBox="0 0 {CW} {CH}" preserveAspectRatio="none" onclick={(e) => player.seek(iScale.indexAt(e.clientX, e.currentTarget as Element))} role="presentation">
          <line x1={iScale.xAt(gIdx)} x2={iScale.xAt(gIdx)} y1={iScale.top} y2={iScale.bottom} class="marker" />
          <path d={iScale.path(iEnt)} class="line data" />
          <path d={iScale.path(iMax)} class="line model" />
          {#each iMax as v, i (i)}
            <circle cx={iScale.xAt(i)} cy={iScale.yAt(v)} r={i === gIdx ? 4 : 2.2} class="dot" class:active={i === gIdx} />
          {/each}
        </svg>
        <div class="xaxis mono faint">
          {#each resp.gammas as g (g)}<span class:cur={g === gamma}>{g}×</span>{/each}
        </div>
        <div class="faint hint">
          At γ = {gamma}: {selHead.curves[gIdx].regime} · H {selHead.curves[gIdx].entropy_bits} bits · eff {selHead.curves[gIdx].eff_k} patterns.
          A head whose regime flips near γ = 1 sits at a phase boundary — the trained temperature is doing real work there.
        </div>
      </div>
    {:else}
      <div class="panel empty"><span class="faint">click a head in the regime map</span></div>
    {/if}
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="hopfieldreal" sections={['hopfieldheads', 'hopfieldread']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ grid: pGrid, census: pCensus, inspect: pInspect, guide: pGuide }}
    actions={{ grid: aGrid, inspect: aInspect }}
  />

  <TransportBar {player} note={resp ? `γ = ${gamma} — γ·β rescales every head's softmax; γ = 1 is the model's own 1/√d_k` : ''} />
{/if}

<style>
  .prompt { flex: 1 1 260px; min-width: 200px; }
  .p-input { width: 100%; font-size: 12px; padding: 5px 8px; }
  .pos-input { width: 58px; font-size: 12px; padding: 5px 6px; }

  .gwrap { display: flex; flex-direction: column; gap: 3px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .grow { display: flex; align-items: center; gap: 6px; }
  .glab { flex: 0 0 30px; text-align: right; font-size: 10px; }
  .gcells { display: grid; gap: 2px; }
  .gcell { aspect-ratio: 1 / 1; border: none; border-radius: 2px; padding: 0; cursor: pointer; min-height: 12px; }
  .gcell.sel { outline: 2px solid var(--chosen, #ffd166); outline-offset: -1px; }
  .gcell:hover { outline: 1.5px solid var(--text); outline-offset: -1.5px; }

  .legend { display: flex; gap: 12px; font-size: 11px; align-items: center; }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }

  .pcol { display: flex; flex-direction: column; gap: 6px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .crow { display: flex; align-items: center; gap: 6px; }
  .cbar { flex: 1 1 auto; display: flex; height: 10px; border-radius: 3px; overflow: hidden; background: var(--bg-2); }
  .hint { font-size: 10.5px; line-height: 1.45; }

  svg { width: 100%; flex: 0 0 auto; min-height: 80px; cursor: crosshair; display: block; }
  .line { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .line.model { stroke: var(--model); }
  .line.data { stroke: var(--data); opacity: 0.95; }
  .marker { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot { fill: var(--model); }
  .dot.active { fill: #fff; }
  .xaxis { display: flex; justify-content: space-between; font-size: 10px; }
  .xaxis .cur { color: var(--model); }
</style>
