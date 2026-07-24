<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import { fetchHealth, fetchLens, type LensResponse } from '../lib/logit/api';

  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import LayerGrid from './logit/LayerGrid.svelte';
  import DepthChart from './logit/DepthChart.svelte';

  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('logit', [
    { id: 'grid', title: 'Lens grid' },
    { id: 'depth', title: 'Code length by depth' },
    { id: 'readout', title: 'Rung readout' }
  ]);

  // The real-model counterpart of the mini-GPT lens: same Player, but the axis
  // is DEPTH, not training time — steps are residual-stream rungs.
  const player = new Player<number>();

  let modelName = $state('gpt2');
  let models = $state<string[]>(['gpt2', 'gpt2-medium', 'gpt2-large', 'Qwen/Qwen2.5-0.5B']);
  let prompt = $state('The Eiffel Tower is in the city of');
  let jlens = $state(true);
  let rollout = $state(0);

  let device = $state<string | null>(null); // null = engine unreachable
  let loading = $state(false);
  let error = $state('');
  let resp = $state<LensResponse | null>(null);
  let selPos = $state(0);

  async function checkHealth() {
    try {
      const h = await fetchHealth();
      device = h.device;
      models = h.models;
      return true;
    } catch {
      device = null;
      return false;
    }
  }

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
      device = device ?? 'up';
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      await checkHealth();
    } finally {
      loading = false;
    }
  }

  // On entry: if the engine is up, run the default prompt so the lens is alive.
  $effect(() => {
    checkHealth().then((ok) => {
      if (ok && !resp) void run();
    });
  });

  function pick(row: number, pos: number) {
    player.seek(row);
    selPos = pos;
  }

  const cur = $derived(player.index);
  const note = $derived(
    resp ? `${resp.layers[cur]} — ${resp.bits[cur]?.toFixed(2)}b for “${resp.pred.token}”` : ''
  );
  const atPred = $derived(resp !== null && selPos === resp.tokens.length - 1);
  const cellTop = $derived(resp ? resp.grid[cur]?.[selPos] ?? [] : []);
  const jTop = $derived(resp?.jtop && atPred ? resp.jtop[cur] : null);

  const barW = (bits: number) =>
    resp ? `${Math.min(100, (bits / (resp.uniform * 1.3)) * 100)}%` : '0%';
</script>

<div class="topbar panel">
  {@render brand()}

  <span class="formula mono" title="per-layer code length of the next token — L(D|M) refined with depth">
    <b style="color:var(--data)">−log₂ p</b> · depth
  </span>

  <label class="f">
    <span class="lbl">model</span>
    <select bind:value={modelName}>
      {#each models as m (m)}<option value={m}>{m}</option>{/each}
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

  <button class="primary" onclick={run} disabled={loading || device === null}>
    {loading ? 'running…' : 'run ▸'}
  </button>

  <span class="status mono" class:off={device === null} title="engine service (engine/, port 5181)">
    {device === null ? 'engine offline' : `engine · ${device}`}
  </span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if device === null && !resp}
  <div class="panel empty">
    <p class="mono">engine offline</p>
    <p class="faint">
      This lens reads a <b>real</b> HuggingFace model through a local engine service. Start it:
    </p>
    <pre class="mono">cd engine
uv run uvicorn main:app --port 5181</pre>
    <p class="faint">
      (set <span class="mono">HF_HOME</span> first to reuse an existing model cache — see README), then
      <button class="ghost" onclick={() => checkHealth().then((ok) => { if (ok) void run(); })}>retry</button>
    </p>
  </div>
{:else if resp}
  <PanelHost manager={panels}>
    <div class="col main">
      <Panel manager={panels} id="grid" weight={2}>
        {#snippet actions()}
          <span class="mono">{resp!.layers.length} rungs × {resp!.tokens.length} positions</span>
        {/snippet}
        <LayerGrid
          layers={resp.layers}
          tokens={resp.tokens}
          grid={resp.grid}
          nPrompt={resp.n_prompt ?? resp.tokens.length}
          index={cur}
          {selPos}
          onPick={pick}
        />
      </Panel>
    </div>

    <div class="col side">
      <Panel manager={panels} id="depth" weight={1}>
        <DepthChart
          bits={resp.bits}
          jbits={resp.jbits}
          uniform={resp.uniform}
          index={cur}
          predToken={resp.pred.token}
          onSeek={(i) => player.seek(i)}
        />
      </Panel>

      <Panel manager={panels} id="readout" weight={1}>
        {#snippet actions()}
          <span class="mono">{resp!.layers[cur]} @ “{resp!.tokens[selPos]}”</span>
        {/snippet}
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
            {#if atPred}
              <div class="bitsline mono faint">
                −log₂ p({resp.pred.token}) = <b style="color:var(--data)">{resp.bits[cur]?.toFixed(2)}b</b>
                <span class="track slim"><span class="fill" style="width:{barW(resp.bits[cur] ?? 0)}; background:var(--data)"></span></span>
              </div>
            {/if}
          </div>

          {#if jTop}
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
                −log₂ p({resp.pred.token}) = <b style="color:var(--model)">{resp.jbits?.[cur]?.toFixed(2)}b</b>
                <span class="track slim"><span class="fill" style="width:{barW(resp.jbits?.[cur] ?? 0)}; background:var(--model)"></span></span>
              </div>
            </div>
          {:else if resp.jtop}
            <div class="ghead mono faint">J-lens readout is at the prediction position — click the last column</div>
          {/if}

          <div class="predline mono faint">
            model's answer: <b>{resp.pred.token}</b> ({(resp.pred.p * 100).toFixed(1)}%,
            {resp.pred.bits.toFixed(2)}b) · uniform {resp.uniform.toFixed(1)}b
          </div>
        </div>
      </Panel>
    </div>
  </PanelHost>

  <div class="panel transport-panel">
    <Controls {player} />
    <div class="note mono" title={note}>{note}{error ? ` · ${error}` : ''}</div>
  </div>
{:else}
  <div class="panel empty"><p class="mono faint">{loading ? 'running…' : error || 'ready'}</p></div>
{/if}

<style>
  .topbar { display: flex; align-items: center; gap: 12px; padding: 8px 12px; flex-wrap: wrap; }
  .formula { font-size: 12px; white-space: nowrap; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 11px; color: var(--muted); }
  .data-input { flex: 1; min-width: 200px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; }
  .rollout { width: 52px; }
  .status { font-size: 10.5px; color: var(--ok, #4dc07d); white-space: nowrap; }
  .status.off { color: var(--bad, #e5484d); }

  .empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
  .empty pre { background: var(--bg-2); border: 1px solid var(--border-2); border-radius: 6px; padding: 10px 14px; font-size: 12px; }

  .col { display: flex; flex-direction: column; gap: 8px; min-height: 0; min-width: 0; }
  .col.main { flex: 3 1 0; }
  .col.side { flex: 2 1 0; }

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

  .transport-panel { display: flex; align-items: center; gap: 16px; padding: 8px 12px; }
  .note { font-size: 11.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
