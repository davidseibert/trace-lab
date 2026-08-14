<script lang="ts">
  import { Player } from '../../lib/player.svelte';
  import { mulberry32 } from '../../lib/llm/rng';
  import type { TensorSnap } from '../../lib/llm/tensor';
  import { computeAttention, randomMatrix, identityMatrix, type AttnConfig } from '../../lib/attn/attention';
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import ActGrid from '../llm/ActGrid.svelte';
  import AttentionView from '../llm/AttentionView.svelte';
  import WeightEditor from './WeightEditor.svelte';
  import CellInspector from './CellInspector.svelte';

  const panels = new PanelManager(
    'attn',
    [
      { id: 'tokens', title: 'Tokens', fit: true },
      { id: 'input', title: 'Input embeddings (X)' },
      { id: 'proj', title: 'Projections — Wq, Wk, Wv, Wo' },
      { id: 'qkv', title: 'Q, K, V' },
      { id: 'scores', title: 'Scores — QKᵀ, scaled' },
      { id: 'weights', title: 'Attention weights' },
      { id: 'output', title: 'Output — per head, concat, final' },
      { id: 'inspect', title: 'Cell inspector' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [
        ['tokens', 'input', 'proj'],
        ['qkv', 'scores'],
        ['weights', 'output', 'inspect', 'guide']
      ],
      widths: [1.1, 1.2, 1.1],
      weights: { proj: 1.6, qkv: 1.3, scores: 1.3, weights: 1.4, output: 1.2 }
    }
  );

  const DMODEL_OPTS = [2, 4, 6, 8];
  const NHEADS_OPTS = [1, 2];
  const PALETTE = ['#ff3355', '#3388ff', '#ffcc00', '#00dd66', '#ff7722', '#bb44ff', '#22bbdd', '#dd4488'];
  const MAX_TOKENS = 6;

  const dParam = router.num('d');
  const hParam = router.num('h');
  let tokenText = $state(router.get('t') ?? 'the cat sat');
  let dModel = $state(dParam !== null && DMODEL_OPTS.includes(dParam) ? dParam : 4);
  let nHeads = $state(hParam !== null && NHEADS_OPTS.includes(hParam) ? hParam : 1);
  let causal = $state(router.bool('c') ?? false);
  let seed = $state(router.num('seed') ?? 1);
  let weightPreset = $state<'random' | 'identity'>(
    router.get('w') === 'identity' ? 'identity' : 'random'
  );

  // Shape + seed + preset live in the URL (hand-edited matrices deliberately
  // don't — they're the one piece of state too heavy for a query string).
  $effect(() => {
    router.setQuery({
      t: tokenText === 'the cat sat' ? null : tokenText,
      d: dModel === 4 ? null : dModel,
      h: nHeads === 1 ? null : nHeads,
      c: causal ? true : null,
      seed: seed === 1 ? null : seed,
      w: weightPreset === 'random' ? null : weightPreset
    });
  });

  const tokens = $derived.by(() => {
    const t = tokenText.trim().split(/\s+/).filter(Boolean).slice(0, MAX_TOKENS);
    return t.length ? t : ['token'];
  });
  const tokenColors = $derived(tokens.map((_, i) => PALETTE[i % PALETTE.length]));

  // Rebuild X and the projection weights whenever their SHAPE or the random
  // seed/preset changes. Deliberately does NOT depend on nHeads or causal —
  // those only change how the existing weights are read, so hand-edits made
  // via WeightEditor survive toggling them.
  let X = $state<TensorSnap>({ data: new Float64Array(0), shape: [0, 0] });
  let Wq = $state<TensorSnap>({ data: new Float64Array(0), shape: [0, 0] });
  let Wk = $state<TensorSnap>({ data: new Float64Array(0), shape: [0, 0] });
  let Wv = $state<TensorSnap>({ data: new Float64Array(0), shape: [0, 0] });
  let Wo = $state<TensorSnap>({ data: new Float64Array(0), shape: [0, 0] });

  $effect(() => {
    const rng = mulberry32(seed);
    const T = tokens.length;
    const d = dModel;
    const mkW = () => (weightPreset === 'identity' ? identityMatrix(d) : randomMatrix(d, d, rng));
    X = randomMatrix(T, d, rng);
    Wq = mkW();
    Wk = mkW();
    Wv = mkW();
    Wo = mkW();
  });

  const cfg = $derived<AttnConfig>({ dModel, nHeads, causal });
  const trace = $derived.by(() => computeAttention(cfg, X, Wq, Wk, Wv, Wo));

  const combinedWeights = $derived.by(() => {
    const H = trace.heads.length;
    const T = trace.T;
    const data = new Float64Array(H * T * T);
    trace.heads.forEach((h, hi) => data.set(h.weights.data, hi * T * T));
    return { data, shape: [H, T, T] };
  });

  // Which query position the arc diagram / weights panel focuses on — scrubbed
  // via the shared Controls transport, reused here as a "which token" walker
  // rather than a training-step walker.
  const player = new Player<number>();
  $effect(() => {
    player.load(tokens.map((_, i) => i));
  });

  let selected = $state<{ head: number; row: number; col: number } | null>(null);
  $effect(() => {
    tokens.length;
    nHeads;
    selected = null;
  });

  function pickCell(head: number, row: number, col: number) {
    selected = { head, row, col };
  }
</script>

<TopBar {panels}>
  <label class="f query">
    <span class="lbl">tokens</span>
    <input
      class="q-input mono"
      bind:value={tokenText}
      spellcheck="false"
      placeholder="the cat sat"
      title={`up to ${MAX_TOKENS} space-separated tokens`}
    />
  </label>

  <div class="f">
    <span class="lbl">d_model</span>
    <div class="toggle-group">
      {#each DMODEL_OPTS as d}
        <button class:active={dModel === d} onclick={() => (dModel = d)}>{d}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl">heads</span>
    <div class="toggle-group">
      {#each NHEADS_OPTS as h}
        <button class:active={nHeads === h} disabled={dModel % h !== 0} onclick={() => (nHeads = h)}>{h}</button>
      {/each}
    </div>
  </div>

  <button class="ghost" class:active={causal} onclick={() => (causal = !causal)} title="Restrict each token to attending only to itself and earlier tokens">
    {causal ? '☑' : '☐'} causal mask
  </button>

  <div class="f">
    <span class="lbl">weights</span>
    <div class="toggle-group">
      <button class:active={weightPreset === 'random'} onclick={() => (weightPreset = 'random')}>random</button>
      <button class:active={weightPreset === 'identity'} onclick={() => (weightPreset = 'identity')} title="Wq = Wk = Wv = Wo = I, so Q = K = V = X">identity</button>
    </div>
  </div>

  <button class="ghost" title="Re-roll random values with a new seed" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  <span class="endspacer"></span>
</TopBar>

{#if trace.T > 0}
{#snippet aTokens()}<span class="mono">{trace.T} tokens · d_model {dModel} · {nHeads} head{nHeads === 1 ? '' : 's'}</span>{/snippet}
{#snippet pTokens()}
  <div class="tokens">
    {#each tokens as t, i}
      <button
        class="tchip mono"
        class:focused={i === player.index}
        style="color:{tokenColors[i]}; border-color:{tokenColors[i]}"
        onclick={() => player.seek(i)}
        title="Focus this token's attention pattern"
      >
        <span class="tpos faint">{i}</span>{t}
      </button>
    {/each}
  </div>
{/snippet}

{#snippet aInput()}<span class="mono">{trace.T} × {dModel} · editable</span>{/snippet}
{#snippet pInput()}
  <WeightEditor value={X} rowLabels={tokens} onChange={(v) => (X = v)} />
{/snippet}

{#snippet aProj()}<span class="mono">{dModel} × {dModel} each</span>{/snippet}
{#snippet pProj()}
  <div class="wgrid scrollbar">
    <div class="wcol">
      <div class="wlabel mono">Wq</div>
      <WeightEditor value={Wq} onChange={(v) => (Wq = v)} />
    </div>
    <div class="wcol">
      <div class="wlabel mono">Wk</div>
      <WeightEditor value={Wk} onChange={(v) => (Wk = v)} />
    </div>
    <div class="wcol">
      <div class="wlabel mono">Wv</div>
      <WeightEditor value={Wv} onChange={(v) => (Wv = v)} />
    </div>
    <div class="wcol">
      <div class="wlabel mono faint">Wo (output proj., auto)</div>
      <ActGrid matrix={Wo} signed />
    </div>
  </div>
{/snippet}

{#snippet aQkv()}<span class="mono">headDim {trace.headDim}</span>{/snippet}
{#snippet pQkv()}
  <div class="heads-col scrollbar">
    {#each trace.heads as h}
      <div class="hgroup">
        {#if trace.heads.length > 1}<div class="head-title faint mono">head {h.head}</div>{/if}
        <div class="triple">
          <div class="qkv-col"><div class="qkv-label mono">Q</div><ActGrid matrix={h.Q} rowLabels={tokens} rowColors={tokenColors} signed /></div>
          <div class="qkv-col"><div class="qkv-label mono">K</div><ActGrid matrix={h.K} rowLabels={tokens} rowColors={tokenColors} signed /></div>
          <div class="qkv-col"><div class="qkv-label mono">V</div><ActGrid matrix={h.V} rowLabels={tokens} rowColors={tokenColors} signed /></div>
        </div>
      </div>
    {/each}
  </div>
{/snippet}

{#snippet aScores()}<span class="mono">click a cell → inspect</span>{/snippet}
{#snippet pScores()}
  <div class="heads-col scrollbar">
    {#each trace.heads as h}
      <div class="hgroup">
        {#if trace.heads.length > 1}<div class="head-title faint mono">head {h.head}</div>{/if}
        <div class="pair">
          <div class="qkv-col"><div class="qkv-label mono">QKᵀ</div><ActGrid matrix={h.scores} rowLabels={tokens} rowColors={tokenColors} colLabel="keys" signed onCellClick={(r, c) => pickCell(h.head, r, c)} /></div>
          <div class="qkv-col"><div class="qkv-label mono">÷ √{trace.headDim}</div><ActGrid matrix={h.scaled} rowLabels={tokens} rowColors={tokenColors} colLabel="keys" signed onCellClick={(r, c) => pickCell(h.head, r, c)} /></div>
        </div>
      </div>
    {/each}
  </div>
{/snippet}

{#snippet aWeights()}<span class="mono">focus: {tokens[player.index]}</span>{/snippet}
{#snippet pWeights()}
  <AttentionView attn={combinedWeights} {tokens} {tokenColors} focusPos={player.index} onCellClick={(h, r, c) => pickCell(h, r, c)} />
{/snippet}

{#snippet aOutput()}<span class="mono">weights · V, then · Wo</span>{/snippet}
{#snippet pOutput()}
  <div class="heads-col scrollbar">
    {#each trace.heads as h}
      <div class="hgroup">
        {#if trace.heads.length > 1}<div class="head-title faint mono">head {h.head} output</div>{/if}
        <ActGrid matrix={h.output} rowLabels={tokens} rowColors={tokenColors} signed />
      </div>
    {/each}
    {#if trace.heads.length > 1}
      <div class="hgroup">
        <div class="head-title faint mono">concat(heads)</div>
        <ActGrid matrix={trace.concat} rowLabels={tokens} rowColors={tokenColors} signed />
      </div>
    {/if}
    <div class="hgroup">
      <div class="head-title faint mono">final = concat · Wo</div>
      <ActGrid matrix={trace.output} rowLabels={tokens} rowColors={tokenColors} signed />
    </div>
  </div>
{/snippet}

{#snippet pInspect()}
  <CellInspector {trace} {tokens} {selected} />
{/snippet}

{#snippet pGuide()}
  <InterpretGuide lens="attn" sections={['attnlab']} />
{/snippet}

<PanelHost
  manager={panels}
  snippets={{
    tokens: pTokens,
    input: pInput,
    proj: pProj,
    qkv: pQkv,
    scores: pScores,
    weights: pWeights,
    output: pOutput,
    inspect: pInspect,
    guide: pGuide
  }}
  actions={{
    tokens: aTokens,
    input: aInput,
    proj: aProj,
    qkv: aQkv,
    scores: aScores,
    weights: aWeights,
    output: aOutput
  }}
/>

<TransportBar {player}>
  scrubbing focuses the arc diagram + weights on token <b>{tokens[player.index]}</b> (position {player.index})
</TransportBar>
{/if}

<style>
  .query { min-width: 160px; }
  .q-input { width: 160px; font-size: 12px; padding: 5px 8px; }
  .endspacer { flex: 0 1 auto; margin-left: auto; }
  button.ghost.active { border-color: var(--model); color: var(--model); }

  .tokens { display: flex; flex-wrap: wrap; gap: 5px; }
  .tchip {
    display: inline-flex; align-items: baseline; gap: 4px;
    font-size: 13px; padding: 3px 7px;
    border: 1px solid; border-radius: 5px; background: var(--bg-2);
  }
  .tchip.focused { background: var(--panel-2); box-shadow: 0 0 0 1.5px currentColor inset; }
  .tpos { font-size: 9px; }

  .wgrid { display: flex; gap: 12px; overflow: auto; min-height: 0; flex: 1 1 auto; align-items: flex-start; }
  .wcol { flex: 1 1 0; min-width: 110px; display: flex; flex-direction: column; gap: 4px; }
  .wlabel { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }

  .heads-col { display: flex; flex-direction: column; gap: 10px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .hgroup { display: flex; flex-direction: column; gap: 4px; }
  .head-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
  .triple, .pair { display: flex; gap: 10px; align-items: flex-start; }
  .qkv-col { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
  .qkv-label { font-size: 10px; color: var(--muted); }
</style>
