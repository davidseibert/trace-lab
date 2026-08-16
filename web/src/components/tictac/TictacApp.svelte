<script lang="ts">
  import { Player } from '../../lib/player.svelte';
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';
  import { trainTrace, type LlmStep, type TrainRun } from '../../lib/llm/trainTrace';
  import {
    analyze,
    boardFromMoves,
    isTerminal,
    legalMoves,
    toMove,
    transformBoard,
    transformMoves,
    winner,
    type Board,
    type CorpusKind
  } from '../../lib/tictac/game';
  import { tictacDataset, TIC_COLORS, TIC_VOCAB } from '../../lib/tictac/dataset';
  import {
    buildProbeSuite,
    computeMetrics,
    equivError,
    gameIds,
    makeReplay,
    pca2,
    policyAt,
    sparsityReport,
    unitFeatureCorrelation,
    SPARSITY_THRESHOLD,
    type MetricsPoint,
    type ProbePosition,
    type Replay
  } from '../../lib/tictac/metrics';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import OutputBars from '../llm/OutputBars.svelte';
  import AttentionView from '../llm/AttentionView.svelte';
  import LensView from '../llm/LensView.svelte';
  import TicBoard from './TicBoard.svelte';
  import BoardHeat from './BoardHeat.svelte';
  import TicMetricsChart from './TicMetricsChart.svelte';
  import EquivPanel, { type EquivItem } from './EquivPanel.svelte';
  import CircuitPanel from './CircuitPanel.svelte';
  import OutcomeStrip from './OutcomeStrip.svelte';

  const panels = new PanelManager(
    'tictac',
    [
      { id: 'board', title: 'Board — click to play' },
      { id: 'dist', title: 'Next-move distribution' },
      { id: 'poset', title: 'Outcomes still on the table' },
      { id: 'metrics', title: 'Training' },
      { id: 'attn', title: 'Attention' },
      { id: 'lens', title: 'Bits ladder' },
      { id: 'equiv', title: 'D₄ equivariance' },
      { id: 'circuit', title: 'Circuit view' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [
        ['board', 'dist', 'poset'],
        ['metrics', 'attn', 'lens'],
        ['equiv', 'circuit', 'guide']
      ],
      widths: [1.1, 1.25, 1.15],
      weights: { board: 1.4, dist: 0.9, poset: 0.8, metrics: 1.2, attn: 1.3, lens: 1.0, equiv: 1.2, circuit: 1.3 }
    }
  );

  const KINDS: CorpusKind[] = ['optimal', 'random', 'mixed'];
  const GAMES_OPTS = [100, 400, 1000];
  const STEPS_OPTS = [200, 400, 1000];
  const L1_OPTS = [0, 0.0003, 0.001, 0.003, 0.01];
  const G_LABELS = ['e', 'r90', 'r180', 'r270', 'flipH', 'flipV', 'diag', 'anti'];

  // ---- settings (URL-backed; defaults elided) -----------------------------
  let kind = $state<CorpusKind>(KINDS.includes(router.get('c') as CorpusKind) ? (router.get('c') as CorpusKind) : 'optimal');
  const nParam = router.num('n');
  const sParam = router.num('steps');
  let nGames = $state(nParam !== null && GAMES_OPTS.includes(nParam) ? nParam : 400);
  let steps = $state(sParam !== null && STEPS_OPTS.includes(sParam) ? sParam : 400);
  let seed = $state(router.num('seed') ?? 1);
  const l1Param = router.num('l1');
  let l1 = $state(l1Param !== null && L1_OPTS.includes(l1Param) ? l1Param : 0.001);
  let thr = $state(router.num('thr') ?? SPARSITY_THRESHOLD);
  let game = $state<number[]>(parseGame(router.get('g') ?? ''));

  function parseGame(s: string): number[] {
    const moves = s.split('').map(Number);
    try {
      boardFromMoves(moves);
      return moves;
    } catch {
      return [];
    }
  }

  $effect(() => {
    router.setQuery({
      c: kind === 'optimal' ? null : kind,
      n: nGames === 400 ? null : nGames,
      steps: steps === 400 ? null : steps,
      seed: seed === 1 ? null : seed,
      l1: l1 === 0.001 ? null : l1,
      thr: thr === SPARSITY_THRESHOLD ? null : thr,
      g: game.length ? game.join('') : null
    });
  });

  // ---- training run (first-paint-then-generate) ---------------------------
  let run = $state<TrainRun | null>(null);
  let replay = $state<Replay | null>(null);
  let suite = $state<ProbePosition[]>([]);
  let mpts = $state<MetricsPoint[]>([]);
  let training = $state(false);

  const player = new Player<LlmStep>();

  const settingsKey = $derived([kind, nGames, steps, seed, l1].join('|'));
  $effect(() => {
    settingsKey;
    training = true;
    setTimeout(() => {
      const ds = tictacDataset(kind, nGames, seed);
      const r = trainTrace(ds, { steps, seed, l1Lambda: l1, evalSample: 50 });
      const s = buildProbeSuite(seed);
      const firstLoad = player.count === 0;
      run = r;
      replay = makeReplay(r);
      suite = s;
      mpts = computeMetrics(r, s, 10);
      // reload keeps the scrub position across retrains (compare λ at step k);
      // the very first load lands on the trained net, not the random init.
      player.reload(r.steps);
      if (firstLoad) player.seek(r.steps.length - 1);
      training = false;
    }, 0);
  });

  // ---- current game, replayed at the scrubbed step ------------------------
  const board = $derived(boardFromMoves(game));
  const gameOver = $derived(isTerminal(board));
  const ids = $derived(gameIds(game));
  const step = $derived(player.index);

  const pos = $derived.by<ProbePosition | null>(() => {
    if (gameOver) return null;
    return { moves: game, board, legal: legalMoves(board), optimal: analyze(board).optimal };
  });

  const viz = $derived(replay && run ? replay.viz(step, ids) : null);
  const pol = $derived(replay && pos ? policyAt(replay.viz, step, pos) : null);
  const modelPick = $derived.by(() => {
    if (!pol || !pos) return null;
    let best = pos.legal[0];
    for (const m of pos.legal) if (pol.pi[m] > pol.pi[best]) best = m;
    return best;
  });

  // Bits the model paid for each played move (policy code length per ply).
  const moveBits = $derived.by(() => {
    if (!replay) return [];
    return game.map((m, k) => {
      const prefix = game.slice(0, k);
      const b = boardFromMoves(prefix);
      const p: ProbePosition = { moves: prefix, board: b, legal: legalMoves(b), optimal: [] };
      const { pi } = policyAt(replay!.viz, step, p);
      return -Math.log2(Math.max(pi[m], 2 ** -10));
    });
  });

  // ---- attention → board heat --------------------------------------------
  let headSel = $state<'mean' | number>('mean');
  const focusPos = $derived(ids.length - 1);
  const attnMass = $derived.by(() => {
    if (!viz) return null;
    const [H, T] = viz.attn.shape;
    const row = new Float64Array(T);
    const heads = headSel === 'mean' ? [...Array(H).keys()] : [headSel as number];
    for (const h of heads)
      for (let j = 0; j < T; j++) row[j] += viz.attn.data[(h * T + focusPos) * T + j] / heads.length;
    const mass = new Float64Array(9);
    for (let j = 1; j < T; j++) mass[game[j - 1]] += row[j];
    return { mass, startMass: row[0] };
  });

  // ---- equivariance view --------------------------------------------------
  const equivItems = $derived.by<EquivItem[]>(() => {
    if (!replay || !pos) return [];
    return [...Array(8).keys()].map((g) => {
      const gb = transformBoard(g, pos.board) as Board;
      const gp: ProbePosition = {
        moves: transformMoves(g, pos.moves),
        board: gb,
        legal: transformMoves(g, pos.legal),
        optimal: transformMoves(g, pos.optimal)
      };
      return {
        g,
        label: G_LABELS[g],
        board: gb,
        pi: policyAt(replay!.viz, step, gp).pi,
        tv: g === 0 ? 0 : equivError(replay!.viz, step, pos, g)
      };
    });
  });
  const equivMean = $derived(
    equivItems.length ? equivItems.slice(1).reduce((a, it) => a + it.tv, 0) / 7 : 0
  );

  // ---- circuit view -------------------------------------------------------
  const corr = $derived(replay && run ? unitFeatureCorrelation(replay.viz, step, suite, run.cfg.ffnHid) : null);
  const sparsity = $derived(run ? sparsityReport(run, run.weights[step], thr) : null);
  const pca = $derived(replay ? pca2(replay.tokenTable(step)) : []);

  const vocabColors = TIC_VOCAB.map((t) => TIC_COLORS[t]);
  const tokens = $derived(ids.map((i) => TIC_VOCAB[i]));
  const tokenColorList = $derived(tokens.map((t) => TIC_COLORS[t]));

  function play(cell: number) {
    if (gameOver || board[cell] !== 0) return;
    game = [...game, cell];
  }

  const metricNow = $derived.by(() => {
    let best: MetricsPoint | null = null;
    for (const p of mpts) if (p.step <= step) best = p;
    return best;
  });

  const note = $derived.by(() => {
    if (training || !run) return 'training the net…';
    const m = metricNow;
    const side = gameOver ? (winner(board) !== 0 ? `${winner(board) === 1 ? 'X' : 'O'} won` : 'draw') : `${toMove(board) === 1 ? 'X' : 'O'} to move`;
    return (
      `step ${step}/${run.steps.length - 1}` +
      (m ? ` · loss ${m.loss.toFixed(2)}b · agree ${(m.agreement * 100).toFixed(0)}% · equiv ${m.equivariance.toFixed(2)} · sparse ${(m.sparsity * 100).toFixed(0)}%` : '') +
      ` — ply ${game.length}, ${side}`
    );
  });
</script>

<TopBar {panels}>
  <div class="f">
    <span class="lbl">corpus</span>
    <div class="toggle-group">
      {#each KINDS as k (k)}
        <button class:active={kind === k} onclick={() => (kind = k)}>{k}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl">games</span>
    <div class="toggle-group">
      {#each GAMES_OPTS as n (n)}
        <button class:active={nGames === n} onclick={() => (nGames = n)}>{n}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl">steps</span>
    <div class="toggle-group">
      {#each STEPS_OPTS as s (s)}
        <button class:active={steps === s} onclick={() => (steps = s)}>{s}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl" title="L1 penalty on all weights — the sparsity pressure that makes the circuit view legible">λ L1</span>
    <div class="toggle-group">
      {#each L1_OPTS as v (v)}
        <button class:active={l1 === v} onclick={() => (l1 = v)}>{v === 0 ? 'off' : v}</button>
      {/each}
    </div>
  </div>

  <button class="ghost" title="Re-roll corpus, init, and probe suite" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  {#if training}<span class="mono trainflag">training…</span>{/if}
  <span class="endspacer"></span>
</TopBar>

{#if !run}
  <div class="panel empty bigwait">
    <p class="mono">training the net…</p>
    <p class="faint">~2 s: {nGames} {kind} games, {steps} updates, λ = {l1}</p>
  </div>
{:else}
  {#snippet aBoard()}<span class="mono">{gameOver ? (winner(board) !== 0 ? `${winner(board) === 1 ? 'X' : 'O'} wins` : 'draw') : `${toMove(board) === 1 ? 'X' : 'O'} to move`}{#if pol} · illegal {(pol.illegalMass * 100).toFixed(0)}%{/if}</span>{/snippet}
  {#snippet pBoard()}
    <div class="pcol">
      <TicBoard {board} pi={pol?.pi ?? null} optimal={pos?.optimal ?? []} {modelPick} onCellClick={gameOver ? undefined : play} />
      <div class="brow">
        <button class="ghost" disabled={!game.length} onclick={() => (game = game.slice(0, -1))}>⌫ undo</button>
        <button class="ghost" disabled={!game.length} onclick={() => (game = [])}>reset</button>
        <button class="ghost" disabled={gameOver || modelPick === null} onclick={() => modelPick !== null && play(modelPick)} title="Play the model's argmax move">▶ model move</button>
      </div>
      <div class="faint hint">blue fill = model policy · green ring = minimax-optimal · yellow outline = model's pick</div>
    </div>
  {/snippet}

  {#snippet pDist()}
    {#if viz && pos}
      <OutputBars probs={viz.probs} vocab={TIC_VOCAB} colors={vocabColors} predId={(modelPick ?? -1) + 1} targetToken={pos.optimal.length ? String(pos.optimal.reduce((a, b) => (viz!.probs[a + 1] >= viz!.probs[b + 1] ? a : b))) : ''} />
      <div class="faint hint">raw distribution — mass on occupied cells and '·' is the model not knowing the rules yet</div>
    {:else}
      <div class="panel empty"><span class="faint">game over — no next move to predict</span></div>
    {/if}
  {/snippet}

  {#snippet pPoset()}
    <OutcomeStrip {game} bits={moveBits} />
  {/snippet}

  {#snippet aMetrics()}{#if metricNow}<span class="mono">every 10 steps · suite of {suite.length}</span>{/if}{/snippet}
  {#snippet pMetrics()}
    <TicMetricsChart points={mpts} {step} onSeek={(s) => player.seek(s)} />
  {/snippet}

  {#snippet aAttn()}
    <span class="mono">
      dst pos {focusPos} ·
      <button class="hbtn" class:on={headSel === 'mean'} onclick={() => (headSel = 'mean')}>mean</button>
      {#each [...Array(run?.cfg.nHeads ?? 0).keys()] as h (h)}
        <button class="hbtn" class:on={headSel === h} onclick={() => (headSel = h)}>h{h}</button>
      {/each}
    </span>
  {/snippet}
  {#snippet pAttn()}
    {#if viz && attnMass}
      <div class="pcol">
        <BoardHeat mass={attnMass.mass} startMass={attnMass.startMass} />
        <div class="faint hint">attention of the latest position, mapped back onto the squares it read</div>
        <AttentionView attn={viz.attn} {tokens} tokenColors={tokenColorList} {focusPos} />
      </div>
    {/if}
  {/snippet}

  {#snippet pLens()}
    {#if replay && modelPick !== null}
      <LensView report={replay.lens(step, ids)} vocab={TIC_VOCAB} colors={vocabColors} focusId={modelPick + 1} />
    {:else}
      <div class="panel empty"><span class="faint">no next move to price</span></div>
    {/if}
  {/snippet}

  {#snippet aEquiv()}<span class="mono">mean TV {equivMean.toFixed(3)}</span>{/snippet}
  {#snippet pEquiv()}
    {#if equivItems.length}
      <div class="pcol">
        <EquivPanel items={equivItems} />
        <div class="faint hint">the same position under all 8 symmetries, each with the model's own policy. A perfectly equivariant net would draw one picture eight ways; the red bars are how far it falls short.</div>
      </div>
    {:else}
      <div class="panel empty"><span class="faint">game over</span></div>
    {/if}
  {/snippet}

  {#snippet aCircuit()}{#if sparsity && run}<span class="mono">{sparsity.liveUnits}/{run.cfg.ffnHid} units live</span>{/if}{/snippet}
  {#snippet pCircuit()}
    {#if corr && sparsity}
      <CircuitPanel {corr} report={sparsity} threshold={thr} onThreshold={(v) => (thr = v)} {pca} vocab={TIC_VOCAB} />
    {/if}
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="tictac" sections={['tictaccircuit', 'tictacequiv']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{
      board: pBoard,
      dist: pDist,
      poset: pPoset,
      metrics: pMetrics,
      attn: pAttn,
      lens: pLens,
      equiv: pEquiv,
      circuit: pCircuit,
      guide: pGuide
    }}
    actions={{
      board: aBoard,
      metrics: aMetrics,
      attn: aAttn,
      equiv: aEquiv,
      circuit: aCircuit
    }}
  />

  <TransportBar {player} {note} converged={player.atEnd} />
{/if}

<style>
  .endspacer { flex: 0 1 auto; margin-left: auto; }
  .trainflag { font-size: 11px; color: var(--data); }
  .bigwait { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; }

  .pcol { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .brow { display: flex; gap: 6px; }
  .hint { font-size: 10.5px; line-height: 1.4; }

  .hbtn {
    font-size: 10px; padding: 1px 6px; margin-left: 2px;
    border: 1px solid var(--border-2); border-radius: 4px;
    background: var(--bg-2); color: var(--muted); cursor: pointer;
  }
  .hbtn.on { color: var(--model); border-color: var(--model); }
</style>
