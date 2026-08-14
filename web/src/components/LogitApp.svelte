<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import {
    fetchColumn,
    fetchLens,
    type ColumnResponse,
    type LensResponse
  } from '../lib/logit/api';
  import { engine } from '../lib/logit/engine.svelte';
  import { router } from '../lib/router.svelte';

  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import EngineOffline from './shell/EngineOffline.svelte';
  import LayerGrid from './logit/LayerGrid.svelte';
  import DepthChart from './logit/DepthChart.svelte';

  const panels = new PanelManager(
    'logit',
    [
      { id: 'grid', title: 'Lens grid', zoomable: true },
      { id: 'depth', title: 'Code length by depth' },
      { id: 'readout', title: 'Rung readout', zoomable: true },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['grid'], ['depth', 'readout', 'guide']],
      widths: [3, 2],
      weights: { grid: 2 }
    }
  );

  // The real-model counterpart of the mini-GPT lens: same Player, but the axis
  // is DEPTH, not training time — steps are residual-stream rungs.
  const player = new Player<number>();

  const DEFAULT_PROMPT = 'The Eiffel Tower is in the city of';
  let modelName = $state(router.get('m') ?? 'gpt2');
  let prompt = $state(router.get('p') ?? DEFAULT_PROMPT);
  let jlens = $state(router.bool('j') ?? true);
  let rollout = $state(router.num('roll') ?? 0);

  // The full run recipe lives in the URL — greedy runs are deterministic, so a
  // copied link IS the run.
  $effect(() => {
    router.setQuery({
      m: modelName === 'gpt2' ? null : modelName,
      p: prompt === DEFAULT_PROMPT ? null : prompt,
      j: jlens ? null : false,
      roll: rollout || null
    });
  });

  // Prompts we set programmatically (defaults, per-model samples). Picking a
  // model swaps the prompt only if the current one is auto — never over a
  // hand-typed prompt. Local checkpoints declare sample prompts that actually
  // fit their vocab/positions ("17+25=", not Eiffel).
  const autoPrompts = new Set([DEFAULT_PROMPT]);
  function pickModel(name: string) {
    modelName = name;
    const want = engine.info(name)?.prompts?.[0] ?? DEFAULT_PROMPT;
    if (autoPrompts.has(prompt.trim()) && prompt.trim() !== want) {
      prompt = want;
      autoPrompts.add(want);
    }
  }
  $effect(() => {
    for (const m of engine.models) for (const p of m.prompts ?? []) autoPrompts.add(p);
  });

  let loading = $state(false);
  let error = $state('');
  let resp = $state<LensResponse | null>(null);
  let selPos = $state(0);
  let col = $state<ColumnResponse | null>(null);
  let colSeq = 0;

  async function run() {
    if (loading || !prompt.trim()) return;
    loading = true;
    error = '';
    try {
      const r = await fetchLens({ model: modelName, prompt, jlens, rollout });
      resp = r;
      selPos = r.tokens.length - 1;
      player.load(r.layers.map((_, i) => i));
      player.seek(r.layers.length - 1); // start at the top rung — scrub down to see it form
      engine.markUp();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      await engine.check();
    } finally {
      loading = false;
    }
  }

  // On entry: if the engine is up, run the restored/default prompt so the lens
  // is alive (deterministic, so a URL-restored run reproduces itself).
  $effect(() => {
    engine.check().then((ok) => {
      if (ok && !resp) void run();
    });
  });

  function pick(row: number, pos: number) {
    player.seek(row);
    selPos = pos;
  }

  // Lazy per-column ladder: /lens carries bits/J only for the last column, so
  // selecting another one asks the engine for that column's ladder (a forward
  // pass plus one JVP per rung). Sequence-guarded so a slow reply can't land
  // on a newer selection.
  $effect(() => {
    const r = resp;
    if (!r) return;
    const p = selPos;
    const seq = ++colSeq;
    col = null;
    if (p === r.tokens.length - 1) return; // /lens already has this ladder
    fetchColumn({
      model: r.model,
      prompt: r.prompt,
      pos: p,
      jlens,
      rollout: r.tokens.length - (r.n_prompt ?? r.tokens.length)
    })
      .then((c) => {
        if (seq === colSeq) col = c;
      })
      .catch(() => {}); // the grid cell still shows its classic top-k
  });

  const cur = $derived(player.index);
  const atPred = $derived(resp !== null && selPos === resp.tokens.length - 1);
  // The selected column's ladder: straight off /lens at the last column, the
  // lazily fetched one elsewhere, null while that fetch is in flight.
  const ladder = $derived(
    !resp
      ? null
      : atPred
        ? { pred: resp.pred, bits: resp.bits, jbits: resp.jbits, jtop: resp.jtop }
        : col && col.pos === selPos
          ? { pred: col.pred, bits: col.bits, jbits: col.jbits, jtop: col.jtop }
          : null
  );
  const note = $derived(
    resp
      ? ladder
        ? `${resp.layers[cur]} — ${ladder.bits[cur]?.toFixed(2)}b for “${ladder.pred.token}”`
        : `${resp.layers[cur]} — computing column ${selPos}…`
      : ''
  );
  const cellTop = $derived(resp ? resp.grid[cur]?.[selPos] ?? [] : []);
  const jTop = $derived(ladder?.jtop ? ladder.jtop[cur] : null);

  const selInfo = $derived(engine.info(modelName));

  const barW = (bits: number) =>
    resp ? `${Math.min(100, (bits / (resp.uniform * 1.3)) * 100)}%` : '0%';
</script>

<TopBar {panels}>
  <span class="formula mono" title="per-layer code length of the next token — L(D|M) refined with depth">
    <b style="color:var(--data)">−log₂ p</b> · depth
  </span>

  <label class="f">
    <span class="lbl">model</span>
    <select value={modelName} onchange={(e) => pickModel((e.currentTarget as HTMLSelectElement).value)}>
      <optgroup label="hub">
        {#each engine.hub as m (m.name)}<option value={m.name}>{m.name}</option>{/each}
      </optgroup>
      {#if engine.local.length}
        <optgroup label="local checkpoints (Train·real)">
          {#each engine.local as m (m.name)}
            <option value={m.name} title={m.note}>{m.name}{m.n_positions ? ` · ${m.n_positions} pos` : ''}</option>
          {/each}
        </optgroup>
      {/if}
      {#if !engine.models.some((m) => m.name === modelName)}
        <option value={modelName}>{modelName}</option>
      {/if}
    </select>
  </label>

  <input
    class="data-input mono"
    type="text"
    bind:value={prompt}
    spellcheck="false"
    placeholder="type a prompt…"
    onkeydown={(e) => e.key === 'Enter' && run()}
  />

  <label class="cb" title="also decode each rung transported through the remaining layers (one JVP per rung)">
    <input type="checkbox" bind:checked={jlens} /> J-lens
  </label>

  <label class="f" title="greedy-decode this many tokens server-side and lens over prompt+continuation — one column per generated token">
    <span class="lbl">rollout</span>
    <input class="rollout mono" type="number" min="0" max="64" bind:value={rollout} />
  </label>

  <button class="primary" onclick={run} disabled={loading || !engine.up}>
    {loading ? 'running…' : 'run ▸'}
  </button>

  {#if selInfo?.note}
    <span class="hint mono" title={selInfo.note}>ℹ {selInfo.note}</span>
  {/if}

  <span class="status mono" class:off={!engine.up} title="engine service (engine/, port 5181)">
    {engine.up ? `engine · ${engine.device}` : 'engine offline'}
  </span>
</TopBar>

{#if !engine.up && !resp}
  <EngineOffline
    what="This lens reads a real HuggingFace model through the local engine service."
    onUp={() => void run()}
  />
{:else if resp}
  {#snippet aGrid()}
    <span class="mono">{resp!.layers.length} rungs × {resp!.tokens.length} positions</span>
  {/snippet}
  {#snippet pGrid()}
    <LayerGrid
      layers={resp!.layers}
      tokens={resp!.tokens}
      grid={resp!.grid}
      nPrompt={resp!.n_prompt ?? resp!.tokens.length}
      index={cur}
      {selPos}
      onPick={pick}
    />
  {/snippet}

  {#snippet pDepth()}
    <DepthChart
      bits={ladder?.bits ?? resp!.bits}
      jbits={ladder?.jbits ?? null}
      uniform={resp!.uniform}
      index={cur}
      predToken={ladder?.pred.token ?? resp!.pred.token}
      onSeek={(i) => player.seek(i)}
    />
  {/snippet}

  {#snippet aReadout()}
    <span class="mono">{resp!.layers[cur]} @ “{resp!.tokens[selPos]}”</span>
  {/snippet}
  {#snippet pReadout()}
    <div class="readout scrollbar">
          <div class="group">
            <div class="ghead mono faint">logit lens — decode the rung as-is</div>
            {#each cellTop as { t, p } (t)}
              <div class="tokrow">
                <span class="tk mono">{t}</span>
                <span class="track"><span class="fill" style="width:{(p * 100).toFixed(1)}%; background:var(--data)"></span></span>
                <span class="pct mono">{(p * 100).toFixed(1)}%</span>
              </div>
            {/each}
            {#if ladder}
              <div class="bitsline mono faint">
                −log₂ p({ladder.pred.token}) = <b style="color:var(--data)">{ladder.bits[cur]?.toFixed(2)}b</b>
                <span class="track slim"><span class="fill" style="width:{barW(ladder.bits[cur] ?? 0)}; background:var(--data)"></span></span>
              </div>
            {/if}
          </div>

          {#if jTop && ladder}
            <div class="group">
              <div class="ghead mono faint">J-lens — transported through the remaining layers first</div>
              {#each jTop as { t, p } (t)}
                <div class="tokrow">
                  <span class="tk mono">{t}</span>
                  <span class="track"><span class="fill" style="width:{(p * 100).toFixed(1)}%; background:var(--model)"></span></span>
                  <span class="pct mono">{(p * 100).toFixed(1)}%</span>
                </div>
              {/each}
              <div class="bitsline mono faint">
                −log₂ p({ladder.pred.token}) = <b style="color:var(--model)">{ladder.jbits?.[cur]?.toFixed(2)}b</b>
                <span class="track slim"><span class="fill" style="width:{barW(ladder.jbits?.[cur] ?? 0)}; background:var(--model)"></span></span>
              </div>
            </div>
          {:else if !ladder}
            <div class="ghead mono faint">computing this column's ladder…</div>
          {/if}

          {#if ladder}
            <div class="predline mono faint">
              after this: <b>{ladder.pred.token}</b> ({(ladder.pred.p * 100).toFixed(1)}%,
              {ladder.pred.bits.toFixed(2)}b) · uniform {resp!.uniform.toFixed(1)}b
            </div>
          {/if}
        </div>
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="logit" sections={['ladder', 'jlens', 'repro', 'small']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ grid: pGrid, depth: pDepth, readout: pReadout, guide: pGuide }}
    actions={{ grid: aGrid, readout: aReadout }}
  />

  <TransportBar {player} note={note + (error ? ` · ${error}` : '')} />
{:else}
  <div class="panel empty"><p class="mono faint">{loading ? 'running…' : error || 'ready'}</p></div>
{/if}

<style>
  .rollout { width: 52px; }
  .hint { max-width: 280px; overflow: hidden; text-overflow: ellipsis; }

  .empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }

  .readout { display: flex; flex-direction: column; gap: 12px; overflow: auto; min-height: 0; }
  .group { display: flex; flex-direction: column; gap: 4px; }
  .ghead { font-size: 10px; letter-spacing: 0.04em; text-transform: uppercase; }
  .tokrow { display: flex; align-items: center; gap: 8px; }
  .tk { min-width: 90px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track { flex: 1; height: 8px; border-radius: 3px; background: var(--bg); overflow: hidden; position: relative; }
  .track.slim { height: 5px; max-width: 110px; }
  .fill { display: block; height: 100%; border-radius: 3px; transition: width 0.15s ease; }
  .pct { font-size: 10px; color: var(--muted); min-width: 44px; text-align: right; }
  .bitsline { display: flex; align-items: center; gap: 8px; font-size: 10.5px; margin-top: 2px; }
  .predline { font-size: 10.5px; margin-top: auto; padding-top: 6px; border-top: 1px solid var(--border-2); }
</style>
