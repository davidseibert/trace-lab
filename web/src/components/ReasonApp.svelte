<script lang="ts">
  import { Player } from '../lib/player.svelte';
  import { PanelManager } from '../lib/panels/panels.svelte';
  import {
    fetchAblate,
    fetchAttn,
    fetchColumn,
    streamChat,
    type AblateResponse,
    type AttnResponse,
    type ColumnResponse,
    type ReasonMeta,
    type ReasonTok
  } from '../lib/logit/api';
  import { engine } from '../lib/logit/engine.svelte';
  import { surprisal, thinkRegions } from '../lib/reason';
  import { router } from '../lib/router.svelte';

  import InterpretGuide from './InterpretGuide.svelte';
  import PanelHost from './PanelHost.svelte';
  import TopBar from './shell/TopBar.svelte';
  import TransportBar from './shell/TransportBar.svelte';
  import EngineOffline from './shell/EngineOffline.svelte';
  import EngineStatus from './shell/EngineStatus.svelte';
  import DepthChart from './logit/DepthChart.svelte';
  import TraceView from './reason/TraceView.svelte';
  import TokenBitsStrip from './reason/TokenBitsStrip.svelte';

  const panels = new PanelManager(
    'reason',
    [
      { id: 'trace', title: 'Reasoning trace', zoomable: true },
      { id: 'tokbits', title: 'Code length per token' },
      { id: 'depth', title: 'Code length by depth' },
      { id: 'readout', title: 'Column readout', zoomable: true },
      { id: 'attn', title: 'Attention' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['trace', 'tokbits'], ['depth', 'readout', 'attn', 'guide']],
      widths: [3, 2],
      weights: { trace: 2 }
    }
  );

  // The player scrubs TOKENS here (the mini-GPT lens scrubs training steps,
  // Logit·real scrubs rungs): playing replays the trace being generated.
  const player = new Player<ReasonTok>();

  const DEFAULT_PROMPT = 'What is 17 * 24? Answer with just the number.';
  let modelName = $state(router.get('m') ?? 'Qwen/Qwen3-0.6B');
  let prompt = $state(router.get('p') ?? DEFAULT_PROMPT);
  let thinking = $state(router.bool('think') ?? true);
  let budget = $state(router.num('budget') ?? 512);
  // Qwen's recommended thinking-mode temperature; 0 = greedy. The engine
  // seeds every sampled run and echoes the seed, so any run can be replayed.
  let temperature = $state(router.num('temp') ?? 0.6);
  let seed = $state<number | null>(router.num('seed'));

  // The six run settings as one query record (defaults → null, kept out of the
  // URL). Both URL writers below build from this; they differ only in
  // provenance — the live effect writes the input boxes, copyRunLink pins what
  // the engine actually ran with (meta.seed / meta.temperature).
  function runQuery(temp: number, seedVal: number | null) {
    return {
      m: modelName === 'Qwen/Qwen3-0.6B' ? null : modelName,
      p: prompt === DEFAULT_PROMPT ? null : prompt,
      think: thinking ? null : false,
      budget: budget === 512 ? null : budget,
      temp: temp === 0.6 ? null : temp,
      seed: seedVal !== null && !Number.isNaN(seedVal) ? seedVal : null
    };
  }

  // Settings — seed and temperature included — live in the URL, so a refresh
  // keeps them and a copied link replays the run.
  $effect(() => {
    router.setQuery(runQuery(temperature, seed));
  });

  let streaming = $state(false);
  let error = $state('');
  let meta = $state<ReasonMeta | null>(null);
  let doneReason = $state<string | null>(null);
  let rung = $state(0); // depth cursor within the selected column
  let abort: AbortController | null = null;

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
            player.append(ev);
          } else if (ev.event === 'done') {
            doneReason = ev.reason;
          } else {
            error = ev.detail;
          }
        },
        abort.signal
      );
      engine.markUp();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        error = e instanceof Error ? e.message : String(e);
        await engine.check();
      }
    } finally {
      streaming = false;
      abort = null;
    }
  }

  function stop() {
    abort?.abort();
  }

  // A finished sampled run is replayable via seed + temp; this builds the URL
  // that pins them (the seed the engine actually used, even when auto).
  let copied = $state(false);
  async function copyRunLink() {
    if (!meta) return;
    // Same record as the live URL, but seed/temp pinned to the finished run.
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(runQuery(meta.temperature, meta.seed))) {
      if (v !== null) q.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
    }
    const qs = q.toString();
    const url = `${location.origin}${location.pathname}#/reason${qs ? `?${qs}` : ''}`;
    try {
      await navigator.clipboard.writeText(url);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      error = 'clipboard blocked — copy the address bar instead';
    }
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

  // ---- Attention: where the selected token's computation looked ----
  // Destination is sel.pos - 1: the column whose forward produced this token
  // (same convention as the J-lens drill-in).
  let shade = $state<'surprisal' | 'attention'>('surprisal');
  let attn = $state<AttnResponse | null>(null);
  let attnLoading = $state(false);
  let headSel = $state<{ layer: number; head: number } | null>(null);
  let headRow = $state<number[] | null>(null);
  let attnSeq = 0;
  const fullIds = () => [...meta!.ids, ...player.steps.map((t) => t.id)];

  $effect(() => {
    const s = sel;
    const on = shade === 'attention';
    headSel = null;
    headRow = null;
    attn = null;
    if (!on || !s || !meta || streaming || s.pos < 2 || s.pos - 1 > 1500) return;
    const seq = ++attnSeq;
    attnLoading = true;
    const t = setTimeout(async () => {
      try {
        const a = await fetchAttn({ model: modelName, ids: fullIds(), pos: s.pos - 1 });
        if (seq === attnSeq) attn = a;
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
      } finally {
        if (seq === attnSeq) attnLoading = false;
      }
    }, 350);
    return () => clearTimeout(t);
  });

  async function pickHead(layer: number, head: number) {
    if (!sel || !meta) return;
    if (headSel && headSel.layer === layer && headSel.head === head) {
      headSel = null;
      headRow = null;
      return;
    }
    headSel = { layer, head };
    try {
      const a = await fetchAttn({ model: modelName, ids: fullIds(), pos: sel.pos - 1, layer, head });
      if (headSel && headSel.layer === layer && headSel.head === head) headRow = a.picked?.vrow ?? null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  // Overlay for TraceView: attention received per position, one shared scale
  // for prompt and trace. Normalized over everything EXCEPT position 0 — the
  // sink would wash out every real signal (it gets its own stat line).
  const attnSrc = $derived(
    shade === 'attention' && attn !== null && sel !== null && attn.pos === sel.pos - 1
      ? (headRow ?? attn.vagg)
      : null
  );
  const attnMax = $derived(attnSrc ? Math.max(1e-9, ...attnSrc.slice(1)) : 1);
  const overlay = $derived.by(() => {
    if (!attnSrc) return null;
    return player.steps.map((t) => Math.min(1, (attnSrc[t.pos] ?? 0) / attnMax));
  });
  // The templated prompt as a dim prefix — so attention into the question
  // (and the <|im_start|> sink) is visible instead of a disembodied stat.
  const prefix = $derived.by(() => {
    if (!meta) return null;
    return meta.tokens.map((t, i) => ({
      t,
      a: attnSrc ? Math.min(1, (attnSrc[i] ?? 0) / attnMax) : 0
    }));
  });
  const promptShare = $derived(
    attn && meta ? attn.vagg.slice(0, meta.n_prompt).reduce((a, b) => a + b, 0) : 0
  );

  // Head grid intensity: focus (inverse entropy), so the pointy heads pop.
  const headFocus = (h: { entropy: number }) =>
    attn ? Math.max(0, 1 - h.entropy / Math.log2(Math.max(2, attn.seq))) : 0;

  // ---- Δbits: re-price the selected token without the think region ----
  let ablate = $state<AblateResponse | null>(null);
  let ablating = $state(false);
  // The think region as sequence POSITIONS for the mask. thinkRegions() finds
  // it robustly (text-space, prompt-seeded — see lib/reason.ts), so this works
  // on models like R1 whose template pre-opens <think>.
  const thinkSpan = $derived.by(() => {
    const span = thinkRegions(player.steps, meta ? meta.tokens.join('') : null).spans[0];
    if (!span) return null;
    return { start: player.steps[span.start]!.pos, end: player.steps[span.end]!.pos };
  });
  $effect(() => {
    void player.index;
    ablate = null;
  });
  async function runAblate() {
    const s = sel;
    if (!s || !meta || !thinkSpan || ablating || s.pos <= thinkSpan.end) return;
    ablating = true;
    try {
      const r = await fetchAblate({
        model: modelName,
        ids: fullIds(),
        pos: s.pos,
        mask_start: thinkSpan.start,
        mask_end: thinkSpan.end + 1
      });
      if (sel === s) ablate = r;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      ablating = false;
    }
  }
  const note = $derived(
    !meta || !sel
      ? streaming
        ? 'generating…'
        : ''
      : `${meta.layers[rung]} @ “${sel.t}” — ${sel.bits[rung]?.toFixed(2)}b` +
        (doneReason ? ` · trace: ${player.count} tokens (${doneReason})` : '')
  );
</script>

<TopBar {panels}>
  <span class="formula mono" title="per-token code length of a reasoning trace — thinking buys the answer down">
    <b style="color:var(--data)">−log₂ p</b> · reasoning
  </span>

  <label class="f">
    <span class="lbl">model</span>
    <select bind:value={modelName}>
      {#each engine.models as m (m.name)}<option value={m.name}>{m.name}</option>{/each}
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
    <button class="primary" onclick={run} disabled={!engine.up}>run ▸</button>
  {/if}

  {#if meta && !streaming}
    <button class="ghost" onclick={copyRunLink} title="Copy a URL that replays this exact run (seed + temperature pinned)">
      {copied ? '✓ copied' : '⎘ run link'}
    </button>
  {/if}

  <EngineStatus />
</TopBar>

{#if !engine.up && !meta}
  <EngineOffline what="This lens streams a reasoning model through the local engine service." />
{:else if meta}
  {#snippet aTrace()}
    <span class="mono">
      {player.count} tokens{streaming ? ' · generating…' : doneReason ? ` · ${doneReason}` : ''}
      {#if meta && meta.temperature > 0}· temp {meta.temperature} seed {meta.seed}{/if}
    </span>
  {/snippet}
  {#snippet pTrace()}
    <TraceView
      steps={player.steps}
      selected={player.index}
      reveal={player.index}
      {overlay}
      {prefix}
      onPick={(i) => player.seek(i)}
    />
  {/snippet}

  {#snippet pTokbits()}
    <TokenBitsStrip
      steps={player.steps}
      selected={player.index}
      prefixText={meta ? meta.tokens.join('') : null}
      onPick={(i) => player.seek(i)}
    />
  {/snippet}

  {#snippet pDepth()}
    {#if sel}
      <DepthChart
        bits={sel.bits}
        jbits={jMatches ? jcol!.jbits : null}
        uniform={meta!.uniform}
        index={rung}
        predToken={sel.t}
        onSeek={(i) => (rung = i)}
      />
    {/if}
  {/snippet}

  {#snippet aReadout()}
    {#if sel}<span class="mono">{meta!.layers[rung]} → “{sel.t}”</span>{/if}
  {/snippet}
  {#snippet pReadout()}
    <div class="readout scrollbar">
          {#if sel}
            <div class="group">
              <div class="ghead mono faint">this token, up the ladder</div>
              <div class="ladderline mono">
                <span class="faint">emitted with</span>
                p = {(sel.p * 100).toFixed(2)}% · <b style="color:var(--data)">{surprisal(sel).toFixed(2)}b</b>
              </div>
              <div class="ladderline mono">
                <span class="faint">{meta!.layers[rung]} says</span>
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
  {/snippet}

  {#snippet aAttn()}
    <div class="toggle-group">
      <button class:active={shade === 'surprisal'} onclick={() => (shade = 'surprisal')}>surprisal</button>
      <button class:active={shade === 'attention'} onclick={() => (shade = 'attention')}>attention</button>
    </div>
  {/snippet}
  {#snippet pAttn()}
    <div class="attnbody scrollbar">
          {#if shade !== 'attention'}
            <div class="ghead mono faint">
              switch the shade to “attention” — the trace then tints by where the
              selected token's computation looked (value-weighted, all heads)
            </div>
          {:else if !sel}
            <div class="ghead mono faint">select a token first</div>
          {:else if attnLoading}
            <div class="ghead mono faint">reading {sel.t} — one forward pass…</div>
          {:else if attn}
            <div class="statline mono faint">
              {headSel ? `L${headSel.layer}·H${headSel.head}` : 'all heads (·‖v‖)'}
              · prompt share {(promptShare * 100).toFixed(0)}%
              · sink {(attn.vagg[0] * 100).toFixed(0)}%
            </div>
            <div class="headgrid" style="--cols: {attn.n_heads}">
              {#each [...attn.heads].sort((a, b) => b.layer - a.layer || a.head - b.head) as h (h.layer * 100 + h.head)}
                <button
                  class="hcell"
                  class:hsel={headSel !== null && headSel.layer === h.layer && headSel.head === h.head}
                  style="--f: {headFocus(h).toFixed(3)}"
                  title={`L${h.layer}·H${h.head} — entropy ${h.entropy.toFixed(1)}b, sink ${(h.sink * 100).toFixed(0)}%`}
                  onclick={() => pickHead(h.layer, h.head)}
                ></button>
              {/each}
            </div>
            <div class="ghead mono faint">rows = layers (bottom row = layer 0) · click a head to isolate its gaze</div>
          {/if}

          {#if thinkSpan && sel && sel.pos > thinkSpan.end}
            <div class="group">
              <div class="ghead mono faint">attention → bits (causal)</div>
              {#if ablate}
                <div class="ladderline mono">
                  “{ablate.token}” costs {ablate.baseline.bits.toFixed(2)}b with the trace,
                  <b style="color:var(--model)">{ablate.masked.bits.toFixed(2)}b</b> without ⟨think⟩
                  — Δ {ablate.delta_bits >= 0 ? '+' : ''}{ablate.delta_bits.toFixed(2)}b
                </div>
                <div class="ladderline mono faint">
                  blind top-3: {#each ablate.masked.top.slice(0, 3) as d (d.t)}<span class="mono">“{d.t}” {(d.p * 100).toFixed(1)}% </span>{/each}
                </div>
              {:else}
                <button class="ghost jbtn" onclick={runAblate} disabled={ablating || streaming}>
                  {ablating ? 'masking the think region + re-forwarding…' : 're-price this token without ⟨think⟩'}
                </button>
              {/if}
            </div>
          {/if}
        </div>
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="reason" sections={['bits', 'ladder', 'jlens', 'attn', 'ablate', 'repro', 'small']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ trace: pTrace, tokbits: pTokbits, depth: pDepth, readout: pReadout, attn: pAttn, guide: pGuide }}
    actions={{ trace: aTrace, readout: aReadout, attn: aAttn }}
  />

  <TransportBar {player} note={note + (error ? ` · ${error}` : '')} />
{:else}
  <div class="panel empty">
    <p class="mono faint">{streaming ? 'generating…' : error || 'ask a question and press run — thinking traces stream in live'}</p>
  </div>
{/if}

<style>
  .budget { width: 64px; }
  .temp { width: 52px; }
  .seedinp { width: 88px; }

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

  .attnbody { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; }
  .statline { font-size: 10.5px; }
  .headgrid {
    display: grid;
    grid-template-columns: repeat(var(--cols), 1fr);
    gap: 2px;
  }
  .hcell {
    all: unset;
    aspect-ratio: 1;
    border-radius: 2px;
    cursor: pointer;
    background: color-mix(in srgb, var(--model) calc(var(--f) * 85%), var(--bg-2));
  }
  .hcell:hover { outline: 1px solid var(--muted); }
  .hcell.hsel { outline: 2px solid var(--data); }
</style>
