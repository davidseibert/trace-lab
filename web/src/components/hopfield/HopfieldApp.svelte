<script lang="ts">
  import { Player } from '../../lib/player.svelte';
  import { mulberry32 } from '../../lib/llm/rng';
  import type { TensorSnap } from '../../lib/llm/tensor';
  import { cellColor } from '../../lib/llm/colors';
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';
  import {
    betaSweep,
    capacityExperiment,
    corrupt,
    logSpacedBetas,
    randomPatterns,
    runClassical,
    runModern,
    separationStats,
    zipTraces,
    type CapacityPoint,
    type ComboStep,
    type Regime
  } from '../../lib/hopfield/hopfield';
  import { FONT, GLYPH_CHARS, GLYPH_COLS, GLYPH_DIM, GLYPH_ROWS, glyphVector, validGlyphChars } from '../../lib/hopfield/glyphs';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import PatternGrid from './PatternGrid.svelte';
  import EnergyChart from './EnergyChart.svelte';
  import BetaSweep from './BetaSweep.svelte';
  import CapacityChart from './CapacityChart.svelte';

  const panels = new PanelManager(
    'hopfield',
    [
      { id: 'patterns', title: 'Stored patterns (X)' },
      { id: 'query', title: 'Query (ξ₀)' },
      { id: 'state', title: 'State ξ — iteration t' },
      { id: 'weights', title: 'Softmax weights over patterns' },
      { id: 'energy', title: 'Energy' },
      { id: 'beta', title: 'Regime vs β' },
      { id: 'capacity', title: 'Capacity' },
      { id: 'bridge', title: '≡ attention' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [
        ['patterns', 'query'],
        ['state', 'weights', 'energy'],
        ['beta', 'capacity', 'bridge', 'guide']
      ],
      widths: [1.0, 1.25, 1.15],
      weights: { patterns: 1.3, query: 1.1, state: 1.5, weights: 1.0, energy: 1.1, beta: 1.1, capacity: 1.1 }
    }
  );

  const N_OPTS = [4, 6, 8, 16];
  const D_OPTS = [16, 64, 256];
  type Mode = 'modern' | 'classical' | 'both';
  const MODES: Mode[] = ['modern', 'classical', 'both'];

  // ---- settings (URL-backed; defaults elided) -----------------------------
  let src = $state<'glyphs' | 'random'>(router.get('src') === 'random' ? 'random' : 'glyphs');
  // AJTX is the well-separated default (Δ = 28): factory settings show crisp
  // retrieval at β*. Add correlated glyphs (A+H, B+P, E+L) to see it degrade.
  let patsText = $state(router.get('pats') ?? 'AJTX');
  const nParam = router.num('n');
  const dParam = router.num('d');
  let nRand = $state(nParam !== null && N_OPTS.includes(nParam) ? nParam : 6);
  let dRand = $state(dParam !== null && D_OPTS.includes(dParam) ? dParam : 64);
  let mode = $state<Mode>(MODES.includes(router.get('mode') as Mode) ? (router.get('mode') as Mode) : 'modern');
  let tgt = $state(router.num('tgt') ?? 0);
  let noise = $state(router.num('noise') ?? 0.15);
  let mask = $state(router.bool('mask') ?? false);
  let seed = $state(router.num('seed') ?? 1);
  /** Absolute β when the slider has been touched; null = the default 1/√d,
   * which then tracks d and stays out of the URL. */
  let betaOverride = $state<number | null>(router.num('beta'));

  $effect(() => {
    router.setQuery({
      src: src === 'glyphs' ? null : src,
      pats: src === 'glyphs' && patsText !== 'AJTX' ? patsText : null,
      n: src === 'random' && nRand !== 6 ? nRand : null,
      d: src === 'random' && dRand !== 64 ? dRand : null,
      mode: mode === 'modern' ? null : mode,
      tgt: tgt === 0 ? null : tgt,
      noise: noise === 0.15 ? null : noise,
      mask: mask ? true : null,
      seed: seed === 1 ? null : seed,
      beta: betaOverride === null ? null : Number(betaOverride.toPrecision(4))
    });
  });

  // ---- patterns ------------------------------------------------------------
  const glyphChars = $derived.by(() => {
    const list = validGlyphChars(patsText);
    return list.length >= 2 ? list : ['A', 'J', 'T', 'X'];
  });

  const X = $derived.by<TensorSnap>(() => {
    if (src === 'glyphs') {
      const data = new Float64Array(glyphChars.length * GLYPH_DIM);
      glyphChars.forEach((ch, i) => data.set(glyphVector(ch), i * GLYPH_DIM));
      return { data, shape: [glyphChars.length, GLYPH_DIM] };
    }
    return randomPatterns(nRand, dRand, mulberry32(seed * 31 + 7));
  });

  const N = $derived(X.shape[0]);
  const d = $derived(X.shape[1]);
  const betaStar = $derived(1 / Math.sqrt(d));
  const beta = $derived(betaOverride ?? betaStar);
  const tgtIdx = $derived(Math.max(0, Math.min(tgt, N - 1)));
  const patLabel = (i: number) => (src === 'glyphs' ? glyphChars[i] : `#${i}`);

  // Display shape: glyphs are 7×5 bitmaps; random ±1 vectors reshape square.
  const gRows = $derived(src === 'glyphs' ? GLYPH_ROWS : Math.round(Math.sqrt(d)));
  const gCols = $derived(src === 'glyphs' ? GLYPH_COLS : Math.round(Math.sqrt(d)));
  const cellPx = $derived(src === 'glyphs' ? 13 : d <= 64 ? 12 : 6);

  const sep = $derived(separationStats(X));

  // ---- query + traces -------------------------------------------------------
  const corrupted = $derived.by(() => {
    const target = X.data.subarray(tgtIdx * d, (tgtIdx + 1) * d) as Float64Array;
    return corrupt(target, mulberry32(seed * 1013 + 51), {
      flipFrac: noise,
      mask: mask ? 'bottom' : 'none'
    });
  });

  // Hand-painted pixels layer on top of the seeded corruption and are the one
  // piece of state not in the URL (same call as Attn·lab's edited matrices).
  let edited = $state<{ modern: Float64Array; classical: Float64Array } | null>(null);
  const settingsKey = $derived([src, glyphChars.join(''), nRand, dRand, tgtIdx, noise, mask, seed].join('|'));
  $effect(() => {
    settingsKey;
    edited = null;
  });

  const xi0 = $derived(edited ?? corrupted);

  function flipPixel(i: number) {
    const m = Float64Array.from(xi0.modern);
    const c = Float64Array.from(xi0.classical);
    const next = m[i] > 0 ? -1 : 1; // 0 (masked) flips to ink
    m[i] = next;
    c[i] = next;
    edited = { modern: m, classical: c };
  }

  const modernTrace = $derived(mode !== 'classical' ? runModern(X, xi0.modern, beta) : null);
  const classicalTrace = $derived(mode !== 'modern' ? runClassical(X, xi0.classical) : null);
  const combo = $derived(zipTraces(modernTrace, classicalTrace));

  const player = new Player<ComboStep>();
  $effect(() => {
    player.reload(combo);
  });

  const cur = $derived(player.current);
  const winner = $derived.by(() => {
    const w = cur?.modern?.weights ?? null;
    if (!w) return cur?.classical ? bestOverlap(cur.classical.overlaps) : null;
    return bestOverlap(w);
  });
  function bestOverlap(v: Float64Array): number {
    let bi = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i;
    return bi;
  }

  const REGIME_VAR: Record<Regime, string> = {
    retrieval: 'var(--good)',
    metastable: 'var(--data)',
    global: 'var(--bad)'
  };

  // ---- β sweep --------------------------------------------------------------
  const sweep = $derived(betaSweep(X, xi0.modern, logSpacedBetas(betaStar)));

  // The slider works in log-multiples of β*; snapping back to exactly β*
  // clears the override so the default keeps tracking d.
  const sliderVal = $derived(Math.log10(beta / betaStar));
  function setSlider(v: number) {
    betaOverride = Math.abs(v) < 0.025 ? null : betaStar * 10 ** v;
  }

  // ---- capacity experiment ----------------------------------------------------
  let capResults = $state<{ points: CapacityPoint[]; stamp: string; d: number } | null>(null);
  let capRunning = $state(false);
  const capStamp = $derived(
    src === 'glyphs'
      ? `glyphs · noise ${noise} · β ${beta.toPrecision(3)} · seed ${seed}`
      : `random d ${dRand} · noise ${noise} · β ${beta.toPrecision(3)} · seed ${seed}`
  );

  function runCapacity() {
    capRunning = true;
    const stamp = capStamp;
    const capD = src === 'glyphs' ? GLYPH_DIM : dRand;
    const Ns = src === 'glyphs' ? [...Array(11).keys()].map((i) => i + 2) : [2, 4, 8, 16, 32, 64, 128];
    const capBeta = beta;
    const capNoise = noise;
    const capSeed = seed;
    const capSrc = src;
    setTimeout(() => {
      const points = capacityExperiment({
        Ns,
        trials: 25,
        noiseFrac: capNoise,
        beta: capBeta,
        seed: capSeed,
        draw: (n, rng) => {
          if (capSrc === 'random') return randomPatterns(n, capD, rng);
          // Glyph trial: a seeded n-subset of the font (partial Fisher–Yates).
          const pool = [...GLYPH_CHARS];
          for (let k = 0; k < n; k++) {
            const r = k + Math.floor(rng() * (pool.length - k));
            [pool[k], pool[r]] = [pool[r], pool[k]];
          }
          const data = new Float64Array(n * GLYPH_DIM);
          for (let i = 0; i < n; i++) data.set(glyphVector(pool[i]), i * GLYPH_DIM);
          return { data, shape: [n, GLYPH_DIM] };
        }
      });
      capResults = { points, stamp, d: capD };
      capRunning = false;
    }, 0);
  }

  function toggleGlyph(ch: string) {
    const has = glyphChars.includes(ch);
    if (has && glyphChars.length <= 2) return; // keep at least two memories
    patsText = has ? glyphChars.filter((c) => c !== ch).join('') : [...glyphChars, ch].join('');
  }

  const note = $derived.by(() => {
    const parts: string[] = [];
    if (modernTrace) {
      const last = modernTrace.steps[modernTrace.steps.length - 1];
      parts.push(
        `modern: ${modernTrace.converged ? `converged in ${modernTrace.steps.length - 1} update${modernTrace.steps.length === 2 ? '' : 's'}` : 'not converged'} (‖Δξ‖ = ${last.deltaNorm.toExponential(1)}) · regime: ${modernTrace.regime}`
      );
    }
    if (classicalTrace)
      parts.push(
        `classical: ${classicalTrace.steps.length - 1} sweep${classicalTrace.steps.length === 2 ? '' : 's'}, ${classicalTrace.steps.reduce((a, s) => a + s.flips, 0)} flips`
      );
    return parts.join(' · ');
  });
  const converged = $derived((modernTrace?.converged ?? true) && (classicalTrace?.converged ?? true));
</script>

<TopBar {panels}>
  <div class="f">
    <span class="lbl">patterns</span>
    <div class="toggle-group">
      <button class:active={src === 'glyphs'} onclick={() => (src = 'glyphs')}>glyphs</button>
      <button class:active={src === 'random'} onclick={() => (src = 'random')}>random ±1</button>
    </div>
  </div>

  {#if src === 'random'}
    <div class="f">
      <span class="lbl">N</span>
      <div class="toggle-group">
        {#each N_OPTS as n (n)}
          <button class:active={nRand === n} onclick={() => (nRand = n)}>{n}</button>
        {/each}
      </div>
    </div>
    <div class="f">
      <span class="lbl">d</span>
      <div class="toggle-group">
        {#each D_OPTS as dv (dv)}
          <button class:active={dRand === dv} onclick={() => (dRand = dv)}>{dv}</button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="f">
    <span class="lbl">mode</span>
    <div class="toggle-group">
      {#each MODES as m (m)}
        <button class:active={mode === m} onclick={() => (mode = m)}>{m}</button>
      {/each}
    </div>
  </div>

  <label class="f beta">
    <span class="lbl">β</span>
    <input
      type="range"
      min="-2"
      max="2"
      step="0.05"
      value={sliderVal}
      oninput={(e) => setSlider(Number((e.currentTarget as HTMLInputElement).value))}
      title="log scale, 0.01×–100× of β* = 1/√d"
    />
    <span class="mono breadout">{beta.toPrecision(3)} ({(beta / betaStar).toPrecision(2)}×β*)</span>
  </label>

  <button class="ghost" title="Re-roll the corruption (and random patterns) with a new seed" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  <span class="endspacer"></span>
</TopBar>

{#snippet aPatterns()}<span class="mono">N {N} · d {d} · Δ {sep.delta.toFixed(0)} ({patLabel(sep.minPair[0])}↔{patLabel(sep.minPair[1])})</span>{/snippet}
{#snippet pPatterns()}
  <div class="pcol scrollbar">
    {#if src === 'glyphs'}
      <div class="chips">
        {#each GLYPH_CHARS as ch (ch)}
          <button class="gchip mono" class:on={glyphChars.includes(ch)} onclick={() => toggleGlyph(ch)} title={glyphChars.includes(ch) ? 'remove from memory' : 'store this pattern'}>{ch}</button>
        {/each}
      </div>
    {/if}
    <div class="pats">
      {#each [...Array(N).keys()] as i (i)}
        <PatternGrid
          pattern={X.data.subarray(i * d, (i + 1) * d)}
          rows={gRows}
          cols={gCols}
          label={patLabel(i)}
          highlight={winner === i}
          cellPx={Math.min(cellPx, 10)}
        />
      {/each}
    </div>
    <div class="faint hint">Δ = worst pattern separation, xᵢᵀxᵢ − max<sub>j≠i</sub> xᵢᵀxⱼ — what the capacity theorems actually charge for.</div>
  </div>
{/snippet}

{#snippet aQuery()}<span class="mono">target {patLabel(tgtIdx)} · {Math.round(noise * 100)}% flips{mask ? ' · masked' : ''}{edited ? ' · hand-edited' : ''}</span>{/snippet}
{#snippet pQuery()}
  <div class="pcol scrollbar">
    <div class="f">
      <span class="lbl">corrupt</span>
      <div class="toggle-group">
        {#each [...Array(N).keys()] as i (i)}
          <button class:active={tgtIdx === i} onclick={() => (tgt = i)}>{patLabel(i)}</button>
        {/each}
      </div>
    </div>
    <label class="f">
      <span class="lbl">noise</span>
      <input type="range" min="0" max="0.5" step="0.05" bind:value={noise} />
      <span class="mono">{Math.round(noise * 100)}%</span>
    </label>
    <button class="ghost" class:active={mask} onclick={() => (mask = !mask)} title="Blank the bottom half: 0 (no information) for the modern state, random ±1 for the classical one">
      {mask ? '☑' : '☐'} mask bottom half
    </button>
    <div class="qrow">
      <PatternGrid pattern={xi0.modern} rows={gRows} cols={gCols} label="ξ₀ (modern) — click to flip" {cellPx} onCellClick={(i) => flipPixel(i)} />
      {#if mode !== 'modern'}
        <PatternGrid pattern={xi0.classical} rows={gRows} cols={gCols} label="ξ₀ (classical)" {cellPx} />
      {/if}
    </div>
  </div>
{/snippet}

{#snippet aState()}<span class="mono">iter {player.index} / {player.count - 1}</span>{/snippet}
{#snippet pState()}
  <div class="srow scrollbar">
    {#if cur?.modern}
      <PatternGrid pattern={cur.modern.xi} rows={gRows} cols={gCols} label={`modern · iter ${Math.min(player.index, (modernTrace?.steps.length ?? 1) - 1)}`} {cellPx} />
    {/if}
    {#if cur?.classical}
      <PatternGrid pattern={cur.classical.xi} rows={gRows} cols={gCols} label={`classical · sweep ${Math.min(player.index, (classicalTrace?.steps.length ?? 1) - 1)}`} {cellPx} />
    {/if}
    <PatternGrid pattern={X.data.subarray(tgtIdx * d, (tgtIdx + 1) * d)} rows={gRows} cols={gCols} label={`target · ${patLabel(tgtIdx)}`} {cellPx} />
    {#if modernTrace && modernTrace.retrieved !== null && modernTrace.retrieved !== tgtIdx}
      <PatternGrid pattern={X.data.subarray(modernTrace.retrieved * d, (modernTrace.retrieved + 1) * d)} rows={gRows} cols={gCols} label={`retrieved · ${patLabel(modernTrace.retrieved)} ⚠`} {cellPx} highlight />
    {/if}
  </div>
{/snippet}

{#snippet aWeights()}
  {#if cur?.modern}
    <span class="mono">
      max {cur.modern.maxWeight.toFixed(2)} · H {cur.modern.entropyBits.toFixed(2)} bits · eff {Math.round(2 ** cur.modern.entropyBits * 10) / 10}
      <span class="regime" style="color:{REGIME_VAR[modernTrace!.regime]}">{modernTrace!.regime}</span>
    </span>
  {/if}
{/snippet}
{#snippet pWeights()}
  <div class="pcol scrollbar">
    {#if cur?.modern}
      <div class="wtitle mono faint">softmax(β·Xξ) — the attention row</div>
      {#each [...Array(N).keys()] as i (i)}
        {@const w = cur.modern.weights[i]}
        <div class="brow">
          <span class="blab mono" class:win={winner === i}>{patLabel(i)}</span>
          <div class="btrack"><div class="bfill" style="width:{(w * 100).toFixed(1)}%"></div></div>
          <span class="bval mono faint">{w.toFixed(3)}</span>
        </div>
      {/each}
    {/if}
    {#if cur?.classical}
      <div class="wtitle mono faint">classical overlaps mᵢ = (1/d)·xᵢᵀξ (not a distribution)</div>
      {#each [...Array(N).keys()] as i (i)}
        {@const m = cur.classical.overlaps[i]}
        <div class="brow">
          <span class="blab mono">{patLabel(i)}</span>
          <div class="btrack"><div class="bfill" class:neg={m < 0} style="width:{(Math.abs(m) * 100).toFixed(1)}%"></div></div>
          <span class="bval mono faint">{m.toFixed(2)}</span>
        </div>
      {/each}
    {/if}
  </div>
{/snippet}

{#snippet pEnergy()}
  <EnergyChart steps={combo} index={player.index} onSeek={(i) => player.seek(i)} />
{/snippet}

{#snippet aBeta()}<span class="mono">β* = 1/√{d} = {betaStar.toPrecision(3)}</span>{/snippet}
{#snippet pBeta()}
  <BetaSweep points={sweep} {betaStar} currentBeta={beta} {N} onPick={(b) => (betaOverride = Math.abs(Math.log10(b / betaStar)) < 0.025 ? null : b)} />
{/snippet}

{#snippet aCapacity()}{#if capResults}<span class="mono faint">{capResults.stamp}</span>{/if}{/snippet}
{#snippet pCapacity()}
  {#if capResults}
    <div class="pcol">
      <CapacityChart points={capResults.points} d={capResults.d} />
      <div class="caprow">
        {#if capResults.stamp !== capStamp}<span class="faint stale">settings changed — results are for: {capResults.stamp}</span>{/if}
        <button class="ghost" disabled={capRunning} onclick={runCapacity}>{capRunning ? 'running…' : '↻ re-run'}</button>
      </div>
    </div>
  {:else}
    <div class="panel empty">
      <div>Sweep N and measure retrieval success (25 trials per point, modern vs classical, identical patterns and corruptions).</div>
      <button class="ghost" disabled={capRunning} onclick={runCapacity}>{capRunning ? 'running…' : '▶ run experiment'}</button>
    </div>
  {/if}
{/snippet}

{#snippet pBridge()}
  <div class="pcol scrollbar bridge">
    <table class="mono">
      <tbody>
        <tr><td>ξ — the state</td><td>q — one query row</td></tr>
        <tr><td>X — {N} stored patterns</td><td>K — the keys</td></tr>
        <tr><td>X again</td><td>V — the values</td></tr>
        <tr><td>β = {beta.toPrecision(3)}</td><td>1/√d_k = {betaStar.toPrecision(3)}</td></tr>
        <tr><td>softmax(β·Xξ)</td><td>one attention row ↓</td></tr>
        <tr><td>ξ_new = Xᵀ·softmax(β·Xξ)</td><td>the attention output row</td></tr>
      </tbody>
    </table>
    {#if cur?.modern}
      <div class="wstrip">
        {#each [...Array(N).keys()] as i (i)}
          <div class="wcell" style="background:{cellColor(cur.modern.weights[i], cur.modern.maxWeight || 1, false)}" title={`${patLabel(i)}: ${cur.modern.weights[i].toFixed(3)}`}></div>
        {/each}
      </div>
    {/if}
    <div class="faint hint">
      Same arithmetic, no analogy — <a href="#/attn">Attn·lab</a> computes this exact row. A transformer learns W_Q/W_K/W_V, i.e. Hopfield retrieval in a learned projected space where keys ≠ values.
    </div>
  </div>
{/snippet}

{#snippet pGuide()}
  <InterpretGuide lens="hopfield" sections={['hopfieldread', 'hopfieldcap']} />
{/snippet}

<PanelHost
  manager={panels}
  snippets={{
    patterns: pPatterns,
    query: pQuery,
    state: pState,
    weights: pWeights,
    energy: pEnergy,
    beta: pBeta,
    capacity: pCapacity,
    bridge: pBridge,
    guide: pGuide
  }}
  actions={{
    patterns: aPatterns,
    query: aQuery,
    state: aState,
    weights: aWeights,
    beta: aBeta,
    capacity: aCapacity
  }}
/>

<TransportBar {player} {note} {converged} />

<style>
  .endspacer { flex: 0 1 auto; margin-left: auto; }
  .beta input[type='range'] { width: 130px; }
  .breadout { font-size: 11px; color: var(--muted); }
  button.ghost.active { border-color: var(--model); color: var(--model); }

  .pcol { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .hint { font-size: 10.5px; line-height: 1.4; }

  .chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .gchip {
    font-size: 12px; padding: 2px 8px;
    border: 1px solid var(--border-2); border-radius: 5px;
    background: var(--bg-2); color: var(--muted); cursor: pointer;
  }
  .gchip.on { color: var(--model); border-color: var(--model); }

  .pats { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start; }
  .qrow, .srow { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; overflow: auto; min-height: 0; }

  .wtitle { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
  .brow { display: flex; align-items: center; gap: 6px; }
  .blab { flex: 0 0 28px; text-align: right; font-size: 11px; }
  .blab.win { color: var(--chosen); }
  .btrack { flex: 1 1 auto; height: 10px; background: var(--bg-2); border-radius: 3px; overflow: hidden; }
  .bfill { height: 100%; background: var(--model); border-radius: 3px; }
  .bfill.neg { background: var(--data); }
  .bval { flex: 0 0 44px; font-size: 10px; text-align: right; }

  .caprow { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .stale { font-size: 10.5px; color: var(--data); }

  .bridge table { border-collapse: collapse; font-size: 11.5px; width: 100%; }
  .bridge td { padding: 4px 8px; border-bottom: 1px solid var(--border-2); }
  .bridge td:first-child { color: var(--model); }
  .bridge td:last-child { color: var(--data); }
  .wstrip { display: flex; gap: 2px; }
  .wcell { flex: 1 1 0; height: 14px; border-radius: 2px; min-width: 8px; }
  .bridge a { color: var(--model); }

  .regime { margin-left: 6px; font-weight: 600; }
</style>
