<script lang="ts">
  import { Player } from '../../lib/player.svelte';
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';
  import type { ForwardViz } from '../../lib/llm/model';
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
  import { TIC_COLORS, TIC_VOCAB } from '../../lib/tictac/dataset';
  import { ticTrain, gameIds, type Arch, type Signal, type TicRun, type TicStep } from '../../lib/tictac/ticTrain';
  import type { BoardViz } from '../../lib/tictac/boardModels';
  import {
    buildProbeSuite,
    computeMetrics,
    equivError,
    klBits,
    pca2,
    policyAt,
    unitFeatureCorrelation,
    SPARSITY_THRESHOLD,
    type MetricsPoint,
    type ProbePosition
  } from '../../lib/tictac/metrics';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import OutputBars from '../llm/OutputBars.svelte';
  import AttentionView from '../llm/AttentionView.svelte';
  import LensView from '../llm/LensView.svelte';
  import ActGrid from '../llm/ActGrid.svelte';
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
  const ARCHS: Arch[] = ['gpt', 'encoder', 'mlp'];
  const SIGNALS: Signal[] = ['games', 'solver', 'distill'];
  const GAMES_OPTS = [100, 400, 1000];
  const STEPS_OPTS = [200, 400, 1000];
  const L1_OPTS = [0, 0.0003, 0.001, 0.003, 0.01];
  const TEMP_OPTS = [1, 2, 4];
  const G_LABELS = ['e', 'r90', 'r180', 'r270', 'flipH', 'flipV', 'diag', 'anti'];

  // ---- settings (URL-backed; defaults elided) -----------------------------
  let kind = $state<CorpusKind>(KINDS.includes(router.get('c') as CorpusKind) ? (router.get('c') as CorpusKind) : 'optimal');
  let arch = $state<Arch>(ARCHS.includes(router.get('a') as Arch) ? (router.get('a') as Arch) : 'gpt');
  const sigParam = router.get('sig') as Signal;
  let signal = $state<Signal>(SIGNALS.includes(sigParam) && !(sigParam === 'distill' && router.get('a') === null) ? sigParam : 'games');
  const tParam = router.num('T');
  let temp = $state(tParam !== null && TEMP_OPTS.includes(tParam) ? tParam : 2);
  const nParam = router.num('n');
  const sParam = router.num('steps');
  let nGames = $state(nParam !== null && GAMES_OPTS.includes(nParam) ? nParam : 400);
  let steps = $state(sParam !== null && STEPS_OPTS.includes(sParam) ? sParam : 400);
  let seed = $state(router.num('seed') ?? 1);
  const l1Param = router.num('l1');
  let l1 = $state(l1Param !== null && L1_OPTS.includes(l1Param) ? l1Param : 0.001);
  let thr = $state(router.num('thr') ?? SPARSITY_THRESHOLD);
  const initialGame = parseGame(router.get('g') ?? '');
  let game = $state<number[]>(initialGame);

  // gpt+distill is self-distillation — not an arm of this grid.
  $effect(() => {
    if (arch === 'gpt' && signal === 'distill') signal = 'games';
  });

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
      a: arch === 'gpt' ? null : arch,
      sig: signal === 'games' ? null : signal,
      T: signal === 'distill' && temp !== 2 ? temp : null,
      n: nGames === 400 ? null : nGames,
      steps: steps === 400 ? null : steps,
      seed: seed === 1 ? null : seed,
      l1: l1 === 0.001 ? null : l1,
      thr: thr === SPARSITY_THRESHOLD ? null : thr,
      g: game.length ? game.join('') : null
    });
  });

  // ---- training runs (first-paint-then-generate, with a teacher cache) -----
  let run = $state<TicRun | null>(null);
  let suite = $state<ProbePosition[]>([]);
  let mpts = $state<MetricsPoint[]>([]);
  let training = $state(false);
  let trainingWhat = $state('');
  let teacherCache: { key: string; run: TicRun } | null = null;
  let teacherRun = $state<TicRun | null>(null);

  const player = new Player<TicStep>();

  const baseKey = $derived([kind, nGames, steps, seed, l1].join('|'));
  const settingsKey = $derived([baseKey, arch, signal, signal === 'distill' ? temp : ''].join('|'));
  $effect(() => {
    settingsKey;
    const key = baseKey;
    const opts = { kind, nGames, steps, seed, l1Lambda: l1 };
    training = true;
    trainingWhat = signal === 'distill' && teacherCache?.key !== key ? 'training teacher…' : 'training…';
    setTimeout(() => {
      let teacher: TicRun | null = null;
      if (signal === 'distill') {
        if (teacherCache?.key !== key) teacherCache = { key, run: ticTrain({ ...opts, arch: 'gpt', signal: 'games' }) };
        teacher = teacherCache.run;
        trainingWhat = 'distilling…';
      }
      const r =
        arch === 'gpt' && signal === 'games' && teacherCache?.key === key
          ? teacherCache.run // the current run IS the teacher config — reuse
          : ticTrain({ ...opts, arch, signal, temperature: temp, teacher: teacher ?? undefined });
      if (arch === 'gpt' && signal === 'games') teacherCache = { key, run: r };
      const s = buildProbeSuite(seed);
      const firstLoad = player.count === 0;
      run = r;
      teacherRun = teacher;
      suite = s;
      mpts = computeMetrics(r, s, 10);
      player.reload(r.steps);
      if (firstLoad) player.seek(r.steps.length - 1);
      training = false;
    }, 0);
  });

  // ---- current game, viewed through the turn cursor -----------------------
  let ply = $state(initialGame.length);
  const viewPly = $derived(Math.min(ply, game.length));
  const viewMoves = $derived(game.slice(0, viewPly));

  const board = $derived(boardFromMoves(viewMoves));
  const gameOver = $derived(isTerminal(board));
  const step = $derived(player.index);

  const pos = $derived.by<ProbePosition | null>(() => {
    if (gameOver) return null;
    return { moves: viewMoves, board, legal: legalMoves(board), optimal: analyze(board).optimal };
  });

  const probs = $derived(run ? run.probsAt(step, viewMoves) : null);
  const pol = $derived(run && pos ? policyAt(run, step, pos) : null);
  const modelPick = $derived.by(() => {
    if (!pol || !pos) return null;
    let best = pos.legal[0];
    for (const m of pos.legal) if (pol.pi[m] > pol.pi[best]) best = m;
    return best;
  });

  const moveBits = $derived.by(() => {
    if (!run) return [];
    return game.map((m, k) => {
      const prefix = game.slice(0, k);
      const b = boardFromMoves(prefix);
      const p: ProbePosition = { moves: prefix, board: b, legal: legalMoves(b), optimal: [] };
      const { pi } = policyAt(run!, step, p);
      return -Math.log2(Math.max(pi[m], 2 ** -10));
    });
  });

  // ---- attention views -----------------------------------------------------
  let headSel = $state<'mean' | number>('mean');
  /** Encoder only: which cell's attention row BoardHeat shows. null = follow
   * the model's pick; click a row in the head grids to inspect any cell —
   * including an empty one (e.g. "what does the block square look at?"). */
  let encFocus = $state<number | null>(null);
  $effect(() => {
    settingsKey;
    viewMoves;
    encFocus = null;
  });
  const viz = $derived(run ? run.vizAt(step, viewMoves) : null);
  const gptViz = $derived(run?.arch === 'gpt' ? (viz as ForwardViz) : null);
  const boardViz = $derived(run && run.arch !== 'gpt' ? (viz as BoardViz) : null);

  const ids = $derived(gameIds(viewMoves));
  const tokens = $derived(ids.map((i) => TIC_VOCAB[i]));
  const tokenColorList = $derived(tokens.map((t) => TIC_COLORS[t]));
  const focusPos = $derived(ids.length - 1);

  /** One attention row (mean or selected head): gpt → the latest position's
   * row mapped onto cells; encoder → the model's PICKED cell's row over cells
   * (the "does the block move attend to the threatened line" readout). */
  const attnHeat = $derived.by(() => {
    if (gptViz) {
      const [H, T] = gptViz.attn.shape;
      const heads = headSel === 'mean' ? [...Array(H).keys()] : [headSel as number];
      const row = new Float64Array(T);
      for (const h of heads)
        for (let j = 0; j < T; j++) row[j] += gptViz.attn.data[(h * T + focusPos) * T + j] / heads.length;
      const mass = new Float64Array(9);
      for (let j = 1; j < T; j++) mass[viewMoves[j - 1]] += row[j];
      return { mass, startMass: row[0], showStart: true };
    }
    const focusCell = encFocus ?? modelPick;
    if (boardViz?.attn && focusCell !== null) {
      const [H] = boardViz.attn.shape;
      const heads = headSel === 'mean' ? [...Array(H).keys()] : [headSel as number];
      const mass = new Float64Array(9);
      for (const h of heads)
        for (let c = 0; c < 9; c++) mass[c] += boardViz.attn.data[(h * 9 + focusCell) * 9 + c] / heads.length;
      return { mass, startMass: 0, showStart: false };
    }
    return null;
  });

  // ---- equivariance view --------------------------------------------------
  const equivItems = $derived.by<EquivItem[]>(() => {
    if (!run || !pos) return [];
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
        pi: policyAt(run!, step, gp).pi,
        tv: g === 0 ? 0 : equivError(run!, step, pos, g)
      };
    });
  });
  const equivMean = $derived(
    equivItems.length ? equivItems.slice(1).reduce((a, it) => a + it.tv, 0) / 7 : 0
  );

  // ---- circuit + distill views --------------------------------------------
  const corr = $derived(run ? unitFeatureCorrelation(run, step, suite) : null);
  const sparsity = $derived(run ? run.sparsity(step, thr) : null);
  const pca = $derived(run ? pca2(run.tokenTable(step)) : []);
  const pcaVocab = $derived(run?.arch === 'gpt' ? TIC_VOCAB : TIC_VOCAB.slice(1));

  /** Mean KL(teacher‖student) over the equivariance sub-suite, final steps —
   * constant per run pair, so computed once. */
  const distillKl = $derived.by(() => {
    if (!run || !teacherRun || run.signal !== 'distill') return null;
    const sub = suite.slice(0, 12);
    if (!sub.length) return null;
    return sub.reduce((a, p) => a + klBits(teacherRun!, run!, p), 0) / sub.length;
  });
  const teacherProbs = $derived(teacherRun && run?.signal === 'distill' ? teacherRun.probsAt(teacherRun.steps.length - 1, viewMoves) : null);

  const vocabColors = TIC_VOCAB.map((t) => TIC_COLORS[t]);

  function play(cell: number) {
    if (gameOver || board[cell] !== 0) return;
    game = [...viewMoves, cell];
    ply = game.length;
  }

  const metricNow = $derived.by(() => {
    let best: MetricsPoint | null = null;
    for (const p of mpts) if (p.step <= step) best = p;
    return best;
  });

  const note = $derived.by(() => {
    if (training || !run) return trainingWhat || 'training…';
    const m = metricNow;
    const side = gameOver ? (winner(board) !== 0 ? `${winner(board) === 1 ? 'X' : 'O'} won` : 'draw') : `${toMove(board) === 1 ? 'X' : 'O'} to move`;
    return (
      `${run.arch}+${run.signal} · step ${step}/${run.steps.length - 1}` +
      (m ? ` · loss ${m.loss.toFixed(2)}b · agree ${(m.agreement * 100).toFixed(0)}% · equiv ${m.equivariance.toFixed(2)} · sparse ${(m.sparsity * 100).toFixed(0)}%` : '') +
      ` — ply ${viewPly}/${game.length}, ${side}`
    );
  });
</script>

<TopBar {panels}>
  <div class="f">
    <span class="lbl">arch</span>
    <div class="toggle-group">
      {#each ARCHS as a (a)}
        <button class:active={arch === a} onclick={() => (arch = a)} title={a === 'gpt' ? 'causal transformer over move sequences — board state must emerge' : a === 'encoder' ? 'bidirectional transformer over the 9 cells — attention can touch empty squares' : 'one hidden layer over the one-hot board — the null model'}>{a}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl">signal</span>
    <div class="toggle-group">
      {#each SIGNALS as s (s)}
        <button
          class:active={signal === s}
          disabled={s === 'distill' && arch === 'gpt'}
          onclick={() => (signal = s)}
          title={s === 'games' ? 'hard targets sampled from the corpus' : s === 'solver' ? 'soft targets: uniform over minimax-optimal moves — the noiseless estimator' : arch === 'gpt' ? 'self-distillation is not an arm of this grid' : 'soft targets from a trained gpt+games teacher'}
        >{s}</button>
      {/each}
    </div>
  </div>

  {#if signal === 'distill'}
    <div class="f">
      <span class="lbl" title="Teacher temperature: q^(1/T) renormalized">T</span>
      <div class="toggle-group">
        {#each TEMP_OPTS as t (t)}
          <button class:active={temp === t} onclick={() => (temp = t)}>{t}</button>
        {/each}
      </div>
    </div>
  {/if}

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

  {#if training}<span class="mono trainflag">{trainingWhat}</span>{/if}
  <span class="endspacer"></span>
</TopBar>

{#if !run}
  <div class="panel empty bigwait">
    <p class="mono">{trainingWhat || 'training…'}</p>
    <p class="faint">{arch}+{signal} · {nGames} {kind} games · {steps} updates · λ = {l1}</p>
  </div>
{:else}
  {#snippet aBoard()}<span class="mono">{gameOver ? (winner(board) !== 0 ? `${winner(board) === 1 ? 'X' : 'O'} wins` : 'draw') : `${toMove(board) === 1 ? 'X' : 'O'} to move`}{#if pol} · illegal {(pol.illegalMass * 100).toFixed(0)}%{/if}</span>{/snippet}
  {#snippet pBoard()}
    <div class="pcol">
      <TicBoard {board} pi={pol?.pi ?? null} optimal={pos?.optimal ?? []} {modelPick} onCellClick={gameOver ? undefined : play} />
      <div class="turnrow">
        <button class="ghost tbtn" disabled={viewPly === 0} onclick={() => (ply = 0)} title="First ply">⏮</button>
        <button class="ghost tbtn" disabled={viewPly === 0} onclick={() => (ply = viewPly - 1)} title="Back one turn">◀</button>
        <input
          class="turnslider"
          type="range"
          min="0"
          max={game.length}
          step="1"
          value={viewPly}
          disabled={!game.length}
          oninput={(e) => (ply = Number((e.currentTarget as HTMLInputElement).value))}
          title="Scrub through the game's turns — every panel follows"
        />
        <button class="ghost tbtn" disabled={viewPly >= game.length} onclick={() => (ply = viewPly + 1)} title="Forward one turn">▶</button>
        <button class="ghost tbtn" disabled={viewPly >= game.length} onclick={() => (ply = game.length)} title="Latest ply">⏭</button>
        <span class="mono faint plylabel">ply {viewPly}/{game.length}</span>
      </div>
      <div class="brow">
        <button class="ghost" disabled={!game.length} onclick={() => { game = game.slice(0, -1); ply = game.length; }}>⌫ undo</button>
        <button class="ghost" disabled={!game.length} onclick={() => { game = []; ply = 0; }}>reset</button>
        <button class="ghost" disabled={gameOver || modelPick === null} onclick={() => modelPick !== null && play(modelPick)} title="Play the model's argmax move">▶ model move</button>
      </div>
      <div class="faint hint">
        {#if viewPly < game.length}viewing ply {viewPly} — click a cell to branch from here · {/if}blue fill = model policy · green ring = minimax-optimal · yellow outline = model's pick
      </div>
    </div>
  {/snippet}

  {#snippet aDist()}{#if distillKl !== null}<span class="mono">KL(teacher‖student) = {distillKl.toFixed(2)} bits</span>{/if}{/snippet}
  {#snippet pDist()}
    {#if probs && pos}
      <div class="pcol">
        <OutputBars {probs} vocab={TIC_VOCAB} colors={vocabColors} predId={(modelPick ?? -1) + 1} targetToken={pos.optimal.length ? String(pos.optimal.reduce((a, b) => (probs![a + 1] >= probs![b + 1] ? a : b))) : ''} />
        {#if teacherProbs}
          <div class="sub mono faint">teacher (gpt+games, final step)</div>
          <OutputBars probs={teacherProbs} vocab={TIC_VOCAB} colors={vocabColors} predId={-1} targetToken="" />
        {:else}
          <div class="faint hint">raw distribution — mass on occupied cells{gptViz ? " and '·'" : ''} is the model not knowing the rules yet</div>
        {/if}
      </div>
    {:else}
      <div class="panel empty"><span class="faint">game over — no next move to predict</span></div>
    {/if}
  {/snippet}

  {#snippet pPoset()}
    <OutcomeStrip {game} bits={moveBits} ply={viewPly} onSeek={(k) => (ply = k)} />
  {/snippet}

  {#snippet aMetrics()}{#if metricNow}<span class="mono">every 10 steps · suite of {suite.length}</span>{/if}{/snippet}
  {#snippet pMetrics()}
    <TicMetricsChart points={mpts} {step} onSeek={(s) => player.seek(s)} />
  {/snippet}

  {#snippet aAttn()}
    {#if run && run.meta.nHeads > 0}
      <span class="mono">
        {run.arch === 'gpt' ? `dst pos ${focusPos}` : encFocus !== null ? `from cell ${encFocus} ✕` : `from picked cell ${modelPick ?? '—'}`}
        {#if encFocus !== null}<button class="hbtn" onclick={() => (encFocus = null)} title="Back to following the model's pick">follow pick</button>{/if} ·
        <button class="hbtn" class:on={headSel === 'mean'} onclick={() => (headSel = 'mean')}>mean</button>
        {#each [...Array(run.meta.nHeads).keys()] as h (h)}
          <button class="hbtn" class:on={headSel === h} onclick={() => (headSel = h)}>h{h}</button>
        {/each}
      </span>
    {/if}
  {/snippet}
  {#snippet pAttn()}
    {#if run?.arch === 'mlp'}
      <div class="panel empty">
        <span class="faint">no attention — that's the point of this arm. The MLP computes the same function; there's just nothing to look at.</span>
      </div>
    {:else if attnHeat && run}
      <div class="pcol">
        <BoardHeat mass={attnHeat.mass} startMass={attnHeat.startMass} showStart={attnHeat.showStart} />
        <div class="faint hint">
          {gptViz
            ? 'attention of the latest position, mapped back onto the squares it read — empty cells are unreachable (no token exists for them)'
            : "one cell's attention row — including edges to and from EMPTY cells, which the gpt arm cannot represent. Click any row in the head grids to inspect that cell."}
        </div>
        {#if gptViz}
          <AttentionView attn={gptViz.attn} {tokens} tokenColors={tokenColorList} {focusPos} />
        {:else if boardViz?.attn}
          <div class="encgrids scrollbar">
            {#each [...Array(run.meta.nHeads).keys()] as h (h)}
              <div class="enccol">
                <div class="sub mono faint">head {h} — cell × cell</div>
                <ActGrid
                  matrix={{ data: boardViz.attn.data.slice(h * 81, (h + 1) * 81), shape: [9, 9] }}
                  rowLabels={TIC_VOCAB.slice(1)}
                  rowColors={TIC_VOCAB.slice(1).map((t) => TIC_COLORS[t])}
                  colLabel="cells"
                  signed={false}
                  onCellClick={(r) => (encFocus = r)}
                />
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {/snippet}

  {#snippet pLens()}
    {#if run && run.arch === 'gpt' && modelPick !== null}
      <LensView report={run.lensAt(step, viewMoves)!} vocab={TIC_VOCAB} colors={vocabColors} focusId={modelPick + 1} />
    {:else if run && run.arch !== 'gpt'}
      <div class="panel empty"><span class="faint">the bits ladder reads MiniGPT residual rungs — gpt arm only</span></div>
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

  {#snippet aCircuit()}{#if sparsity && run}<span class="mono">{sparsity.liveUnits}/{run.meta.nUnits} units live</span>{/if}{/snippet}
  {#snippet pCircuit()}
    {#if corr && sparsity}
      <CircuitPanel {corr} report={sparsity} threshold={thr} onThreshold={(v) => (thr = v)} {pca} vocab={pcaVocab} />
    {/if}
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="tictac" sections={['tictacarch', 'tictaccircuit', 'tictacequiv']} />
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
      dist: aDist,
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
  .turnrow { display: flex; align-items: center; gap: 4px; max-width: 300px; }
  .tbtn { padding: 2px 7px; font-size: 11px; }
  .turnslider { flex: 1 1 auto; min-width: 60px; }
  .plylabel { font-size: 10.5px; white-space: nowrap; }
  .hint { font-size: 10.5px; line-height: 1.4; max-width: 320px; overflow-wrap: break-word; }
  .sub { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }

  .encgrids { display: flex; gap: 12px; overflow: auto; min-height: 0; }
  .enccol { flex: 1 1 0; min-width: 140px; display: flex; flex-direction: column; gap: 4px; }

  .hbtn {
    font-size: 10px; padding: 1px 6px; margin-left: 2px;
    border: 1px solid var(--border-2); border-radius: 4px;
    background: var(--bg-2); color: var(--muted); cursor: pointer;
  }
  .hbtn.on { color: var(--model); border-color: var(--model); }
</style>
