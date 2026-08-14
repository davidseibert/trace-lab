<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Player } from '../lib/player.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import {
    fetchColumn,
    fetchHealth,
    streamChat,
    type ColumnResponse,
    type ReasonMeta,
    type ReasonTok
  } from '../lib/logit/api';

  import Controls from './Controls.svelte';
  import Panel from './Panel.svelte';
  import PanelHost from './PanelHost.svelte';
  import DepthChart from './logit/DepthChart.svelte';
  import TraceView from './reason/TraceView.svelte';
  import TokenBitsStrip from './reason/TokenBitsStrip.svelte';

  let { brand }: { brand: Snippet } = $props();

  const panels = new PanelManager('reason', [
    { id: 'trace', title: 'Reasoning trace' },
    { id: 'tokbits', title: 'Code length per token' },
    { id: 'depth', title: 'Code length by depth' },
    { id: 'readout', title: 'Column readout' }
  ]);

  // The player scrubs TOKENS here (the mini-GPT lens scrubs training steps,
  // Logit·real scrubs rungs): playing replays the trace being generated.
  const player = new Player<ReasonTok>();

  let modelName = $state('Qwen/Qwen3-0.6B');
  let models = $state<string[]>(['Qwen/Qwen3-0.6B']);
  let prompt = $state('What is 17 * 24? Answer with just the number.');
  let thinking = $state(true);
  let budget = $state(512);
  // Qwen's recommended thinking-mode temperature; 0 = greedy. The engine
  // seeds every sampled run and echoes the seed, so any run can be replayed.
  let temperature = $state(0.6);
  let seed = $state<number | null>(null);

  let device = $state<string | null>(null);
  let streaming = $state(false);
  let error = $state('');
  let meta = $state<ReasonMeta | null>(null);
  let doneReason = $state<string | null>(null);
  let rung = $state(0); // depth cursor within the selected column
  let abort: AbortController | null = null;

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
  $effect(() => {
    void checkHealth();
  });

  async function run() {
    if (streaming || !prompt.trim()) return;
    streaming = true;
    error = '';
    meta = null;
    doneReason = null;
    jcol = null;
    player.load([]);
    abort = new AbortController();
    try {
      await streamChat(
        {
          model: modelName,
          prompt,
          thinking,
          max_new: budget,
          temperature,
          ...(seed !== null && !Number.isNaN(seed) ? { seed } : {})
        },
        (ev) => {
          if (ev.event === 'meta') {
            meta = ev;
            rung = ev.layers.length - 1; // start at the top rung, like Logit·real
          } else if (ev.event === 'tok') {
            const follow = player.atEnd;
            player.steps.push(ev);
            if (follow) player.index = player.steps.length - 1;
          } else if (ev.event === 'done') {
            doneReason = ev.reason;
          } else {
            error = ev.detail;
          }
        },
        abort.signal
      );
      device = device ?? 'up';
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        error = e instanceof Error ? e.message : String(e);
        await checkHealth();
      }
    } finally {
      streaming = false;
      abort = null;
    }
  }

  function stop() {
    abort?.abort();
  }

  const sel = $derived(player.current ?? null);

  // J-lens drill-in for the selected column — explicitly requested, never
  // automatic: it re-decodes the trace and runs a JVP per rung (~20s+).
  let jcol = $state<ColumnResponse | null>(null);
  let jloading = $state(false);
  $effect(() => {
    void player.index; // any selection change invalidates the drill-in
    jcol = null;
  });
  async function fetchJ() {
    const s = sel;
    if (!s || jloading || !meta) return;
    jloading = true;
    try {
      // Hand the exact streamed sequence over: fork-proof, and measures the
      // ladder against the token actually taken (matters for sampled traces).
      const c = await fetchColumn({
        model: modelName,
        prompt,
        ids: [...meta.ids, ...player.steps.map((t) => t.id)],
        pos: s.pos - 1, // the column that produced the selected token
        jlens: true
      });
      if (sel === s) jcol = c;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      jloading = false;
    }
  }

  const jMatches = $derived(jcol !== null && sel !== null && jcol.pos === sel.pos - 1);
  const surp = (s: ReasonTok) => -Math.log2(Math.max(s.p, 1e-30));
  const note = $derived(
    !meta || !sel
      ? streaming
        ? 'generating…'
        : ''
      : `${meta.layers[rung]} @ “${sel.t}” — ${sel.bits[rung]?.toFixed(2)}b` +
        (doneReason ? ` · trace: ${player.count} tokens (${doneReason})` : '')
  );
</script>

<div class="topbar panel">
  {@render brand()}

  <span class="formula mono" title="per-token code length of a reasoning trace — thinking buys the answer down">
    <b style="color:var(--data)">−log₂ p</b> · reasoning
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
    placeholder="ask a question…"
    onkeydown={(e) => e.key === 'Enter' && run()}
  />

  <label class="cb" title="open a <think> block via the chat template (Qwen3)">
    <input type="checkbox" bind:checked={thinking} /> thinking
  </label>

  <label class="f" title="token budget: generation stops at EOS or here">
    <span class="lbl">budget</span>
    <input class="budget mono" type="number" min="16" max="2048" step="16" bind:value={budget} />
  </label>

  <label class="f" title="0 = greedy (deterministic but ruminates); ~0.6 recommended for thinking. Bits always price the model's true distribution.">
    <span class="lbl">temp</span>
    <input class="temp mono" type="number" min="0" max="2" step="0.1" bind:value={temperature} />
  </label>

  <label class="f" title="blank = fresh seed each run (shown after); set one to replay a run exactly">
    <span class="lbl">seed</span>
    <input class="seedinp mono" type="number" min="0" placeholder="auto" bind:value={seed} />
  </label>

  {#if streaming}
    <button class="primary" onclick={stop}>stop ■</button>
  {:else}
    <button class="primary" onclick={run} disabled={device === null}>run ▸</button>
  {/if}

  <span class="status mono" class:off={device === null} title="engine service (engine/, port 5181)">
    {device === null ? 'engine offline' : `engine · ${device}`}
  </span>

  {#if panels.isDirty}
    <button class="ghost reset-layout" onclick={() => panels.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

{#if device === null && !meta}
  <div class="panel empty">
    <p class="mono">engine offline</p>
    <p class="faint">This lens streams a reasoning model through a local engine service. Start it:</p>
    <pre class="mono">make up</pre>
    <p class="faint">
      then <button class="ghost" onclick={() => checkHealth()}>retry</button>
    </p>
  </div>
{:else if meta}
  <PanelHost manager={panels}>
    <div class="col main">
      <Panel manager={panels} id="trace" weight={2}>
        {#snippet actions()}
          <span class="mono">
            {player.count} tokens{streaming ? ' · generating…' : doneReason ? ` · ${doneReason}` : ''}
            {#if meta && meta.temperature > 0}· temp {meta.temperature} seed {meta.seed}{/if}
          </span>
        {/snippet}
        <TraceView
          steps={player.steps}
          selected={player.index}
          reveal={player.index}
          onPick={(i) => player.seek(i)}
        />
      </Panel>
      <Panel manager={panels} id="tokbits" weight={1}>
        <TokenBitsStrip steps={player.steps} selected={player.index} onPick={(i) => player.seek(i)} />
      </Panel>
    </div>

    <div class="col side">
      <Panel manager={panels} id="depth" weight={1}>
        {#if sel}
          <DepthChart
            bits={sel.bits}
            jbits={jMatches ? jcol!.jbits : null}
            uniform={meta.uniform}
            index={rung}
            predToken={sel.t}
            onSeek={(i) => (rung = i)}
          />
        {/if}
      </Panel>

      <Panel manager={panels} id="readout" weight={1}>
        {#snippet actions()}
          {#if sel}<span class="mono">{meta!.layers[rung]} → “{sel.t}”</span>{/if}
        {/snippet}
        <div class="readout scrollbar">
          {#if sel}
            <div class="group">
              <div class="ghead mono faint">this token, up the ladder</div>
              <div class="ladderline mono">
                <span class="faint">emitted with</span>
                p = {(sel.p * 100).toFixed(2)}% · <b style="color:var(--data)">{surp(sel).toFixed(2)}b</b>
              </div>
              <div class="ladderline mono">
                <span class="faint">{meta.layers[rung]} says</span>
                “{sel.rtop[rung]}” · {sel.bits[rung]?.toFixed(2)}b for “{sel.t}”
              </div>
            </div>

            <div class="group">
              <div class="ghead mono faint">J-lens — transported through the remaining layers</div>
              {#if jMatches && jcol!.jtop}
                {#each jcol!.jtop[rung] ?? [] as { t, p } (t)}
                  <div class="tokrow">
                    <span class="tk mono">{t}</span>
                    <span class="track"><span class="fill" style="width:{(p * 100).toFixed(1)}%; background:var(--model)"></span></span>
                    <span class="pct mono">{(p * 100).toFixed(1)}%</span>
                  </div>
                {/each}
                <div class="ladderline mono faint">
                  −log₂ p({sel.t}) = <b style="color:var(--model)">{jcol!.jbits?.[rung]?.toFixed(2)}b</b>
                </div>
              {:else}
                <button class="ghost jbtn" onclick={fetchJ} disabled={jloading || streaming}>
                  {jloading ? 'computing (re-decodes the trace + one JVP per rung)…' : 'J-lens this column'}
                </button>
              {/if}
            </div>
          {:else}
            <div class="ghead mono faint">{streaming ? 'waiting for the first token…' : 'no trace yet'}</div>
          {/if}
        </div>
      </Panel>
    </div>
  </PanelHost>

  <div class="panel transport-panel">
    <Controls {player} />
    <div class="note mono" title={note}>{note}{error ? ` · ${error}` : ''}</div>
  </div>
{:else}
  <div class="panel empty">
    <p class="mono faint">{streaming ? 'generating…' : error || 'ask a question and press run — thinking traces stream in live'}</p>
  </div>
{/if}

<style>
  .topbar { display: flex; align-items: center; gap: 12px; padding: 8px 12px; flex-wrap: wrap; }
  .formula { font-size: 12px; white-space: nowrap; }
  .f { display: flex; align-items: center; gap: 6px; }
  .lbl { font-size: 11px; color: var(--muted); }
  .data-input { flex: 1; min-width: 200px; }
  .cb { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); white-space: nowrap; }
  .budget { width: 64px; }
  .temp { width: 52px; }
  .seedinp { width: 88px; }
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
  .ladderline { font-size: 11.5px; display: flex; gap: 6px; align-items: baseline; flex-wrap: wrap; }
  .tokrow { display: flex; align-items: center; gap: 8px; }
  .tk { min-width: 90px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track { flex: 1; height: 8px; border-radius: 3px; background: var(--bg); overflow: hidden; position: relative; }
  .fill { display: block; height: 100%; border-radius: 3px; }
  .pct { font-size: 10px; color: var(--muted); min-width: 44px; text-align: right; }
  .jbtn { font-size: 11px; align-self: flex-start; }

  .transport-panel { display: flex; align-items: center; gap: 16px; padding: 8px 12px; }
  .note { font-size: 11.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
