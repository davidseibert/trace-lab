<script lang="ts">
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { router } from '../../lib/router.svelte';
  import { engine } from '../../lib/logit/engine.svelte';
  import { analyze, boardFromMoves, legalMoves, winner, type CorpusKind } from '../../lib/tictac/game';
  import { ticTrain, type Arch, type Signal, type TicRun } from '../../lib/tictac/ticTrain';
  import { buildProbeSuite, type ProbePosition } from '../../lib/tictac/metrics';
  import {
    defaultLlmMode,
    llmPlayer,
    randomPlayer,
    solverPlayer,
    toyPlayer,
    type TicPlayer
  } from '../../lib/tictac/players';
  import {
    playMatch,
    reportCard,
    roundRobin,
    type GameRecord,
    type MatchResult,
    type ReportCard
  } from '../../lib/tictac/arena';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import EngineStatus from '../shell/EngineStatus.svelte';
  import TicBoard from './TicBoard.svelte';
  import MatchTable from './MatchTable.svelte';
  import OutcomeStrip from './OutcomeStrip.svelte';

  const panels = new PanelManager(
    'arena',
    [
      { id: 'roster', title: 'Roster' },
      { id: 'match', title: 'Duel' },
      { id: 'table', title: 'Round-robin' },
      { id: 'replay', title: 'Replay' },
      { id: 'report', title: 'Report card — vs the solved game' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [
        ['roster', 'match'],
        ['table', 'replay'],
        ['report', 'guide']
      ],
      widths: [1.05, 1.25, 1.2],
      weights: { roster: 1.3, match: 1.2, table: 1.1, replay: 1.3, report: 1.3 }
    }
  );

  const KINDS: CorpusKind[] = ['optimal', 'random', 'mixed'];
  const ARCHS: Arch[] = ['gpt', 'encoder', 'mlp'];
  const SIGNALS: Signal[] = ['games', 'solver', 'distill'];
  const N_OPTS = [1, 10, 25];
  const TEMP_OPTS = [0, 0.5, 1];
  const DEFAULT_ROSTER = ['oracle:perfect', 'oracle:traps', 'oracle:random', 'toy:gpt:games'];
  // Toy training settings are pinned to Tic·tac's defaults; kind/seed are knobs.
  const TOY_GAMES = 400;
  const TOY_STEPS = 400;
  const TOY_L1 = 0.001;

  // ---- settings (URL-backed; defaults elided) -----------------------------
  let kind = $state<CorpusKind>(KINDS.includes(router.get('c') as CorpusKind) ? (router.get('c') as CorpusKind) : 'optimal');
  let seed = $state(router.num('seed') ?? 1);
  let roster = $state<string[]>(parseRoster(router.get('roster')));
  let xId = $state(router.get('x') ?? 'toy:gpt:games');
  let oId = $state(router.get('o') ?? 'oracle:perfect');
  const nParam = router.num('n');
  let nGames = $state(nParam !== null && N_OPTS.includes(nParam) ? nParam : 10);
  const tParam = router.num('t');
  let temp = $state(tParam !== null && TEMP_OPTS.includes(tParam) ? tParam : 0);

  function parseRoster(s: string | null): string[] {
    if (!s) return [...DEFAULT_ROSTER];
    const ids = s.split(',').filter((id) => /^(oracle|toy|llm):/.test(id));
    return ids.length ? ids : [...DEFAULT_ROSTER];
  }

  $effect(() => {
    router.setQuery({
      c: kind === 'optimal' ? null : kind,
      seed: seed === 1 ? null : seed,
      roster: roster.join(',') === DEFAULT_ROSTER.join(',') ? null : roster.join(','),
      x: xId === 'toy:gpt:games' ? null : xId,
      o: oId === 'oracle:perfect' ? null : oId,
      n: nGames === 10 ? null : nGames,
      t: temp === 0 ? null : temp
    });
  });

  // ---- adders --------------------------------------------------------------
  let toyArch = $state<Arch>('encoder');
  let toySignal = $state<Signal>('games');
  let llmModel = $state('Qwen/Qwen3-0.6B');
  let llmMode = $state<'chat' | 'raw' | 'auto'>('auto');

  function addToRoster(id: string) {
    if (!roster.includes(id)) roster = [...roster, id];
  }
  function removeFromRoster(id: string) {
    if (roster.length > 2) roster = roster.filter((r) => r !== id);
  }

  const playerLabel = (id: string): string => {
    if (id === 'oracle:perfect') return 'solver';
    if (id === 'oracle:traps') return 'solver·trappy';
    if (id === 'oracle:random') return 'random';
    if (id.startsWith('toy:')) return id.slice(4).replace(':', '+');
    if (id.startsWith('llm:')) {
      const rest = id.slice(4);
      const li = rest.lastIndexOf(':');
      return `${rest.slice(0, li).split('/').pop()} (${rest.slice(li + 1)})`;
    }
    return id;
  };

  // ---- player construction (toys train on demand, cached) -----------------
  const toyCache = new Map<string, TicRun>();
  const llmCache = new Map<string, TicPlayer>(); // keeps each player's position memo alive
  let busy = $state<string | null>(null);
  let errMsg = $state('');

  const yield0 = () => new Promise((r) => setTimeout(r, 0));

  async function ensureToyRun(arch: Arch, signal: Signal): Promise<TicRun> {
    const key = [arch, signal, kind, seed].join('|');
    let run = toyCache.get(key);
    if (run) return run;
    let teacher: TicRun | undefined;
    if (signal === 'distill') {
      busy = `training teacher for ${arch}+distill…`;
      await yield0();
      teacher = await ensureToyRun('gpt', 'games');
    }
    busy = `training ${arch}+${signal}…`;
    await yield0();
    run = ticTrain({ arch, signal, kind, nGames: TOY_GAMES, steps: TOY_STEPS, seed, l1Lambda: TOY_L1, temperature: 2, teacher });
    toyCache.set(key, run);
    return run;
  }

  async function buildPlayer(id: string): Promise<TicPlayer> {
    if (id === 'oracle:perfect') return solverPlayer('uniform');
    if (id === 'oracle:traps') return solverPlayer('traps');
    if (id === 'oracle:random') return randomPlayer();
    if (id.startsWith('toy:')) {
      const [, arch, signal] = id.split(':') as [string, Arch, Signal];
      return toyPlayer(arch, signal, await ensureToyRun(arch, signal));
    }
    if (id.startsWith('llm:')) {
      let p = llmCache.get(id);
      if (!p) {
        const rest = id.slice(4);
        const li = rest.lastIndexOf(':');
        p = llmPlayer(rest.slice(0, li), rest.slice(li + 1) as 'chat' | 'raw');
        llmCache.set(id, p);
      }
      return p;
    }
    throw new Error(`unknown player ${id}`);
  }

  // ---- duel ----------------------------------------------------------------
  let match = $state<MatchResult | null>(null);
  let selectedGame = $state<GameRecord | null>(null);
  let replayPly = $state(0);

  async function runMatch() {
    if (busy) return;
    errMsg = '';
    try {
      const [pX, pO] = [await buildPlayer(xId), await buildPlayer(oId)];
      busy = 'playing…';
      const m = await playMatch(pX, pO, {
        games: nGames,
        seed,
        temperature: temp,
        onGame: (i) => (busy = `playing… game ${i + 1}/${nGames}`)
      });
      match = m;
      pickGame(m.games[0] ?? null);
    } catch (e) {
      errMsg = e instanceof Error ? e.message : String(e);
    }
    busy = null;
  }

  function pickGame(g: GameRecord | null) {
    selectedGame = g;
    replayPly = 0;
  }

  // ---- round-robin ----------------------------------------------------------
  let tableResults = $state<MatchResult[]>([]);

  async function runTable() {
    if (busy) return;
    errMsg = '';
    try {
      const players: TicPlayer[] = [];
      for (const id of roster) players.push(await buildPlayer(id));
      const results = await roundRobin(players, { games: nGames, seed, temperature: temp }, (done, total, cur) => {
        busy = `round-robin ${done}/${total} · ${cur}`;
      });
      tableResults = results;
    } catch (e) {
      errMsg = e instanceof Error ? e.message : String(e);
    }
    busy = null;
  }

  // ---- report cards ---------------------------------------------------------
  let cards = $state<{ id: string; card: ReportCard }[]>([]);

  async function runReport() {
    if (busy) return;
    errMsg = '';
    try {
      const suite: ProbePosition[] = buildProbeSuite(seed);
      const out: { id: string; card: ReportCard }[] = [];
      for (const id of roster) {
        const p = await buildPlayer(id);
        const card = await reportCard(p, suite, (done, total) => {
          busy = `report ${playerLabel(id)} · ${done}/${total}`;
        });
        out.push({ id, card });
        cards = [...out];
      }
    } catch (e) {
      errMsg = e instanceof Error ? e.message : String(e);
    }
    busy = null;
  }

  // ---- replay derived -------------------------------------------------------
  const rMoves = $derived(selectedGame?.moves ?? []);
  const rPly = $derived(Math.min(replayPly, rMoves.length));
  const rBoard = $derived(boardFromMoves(rMoves.slice(0, rPly)));
  const rPi = $derived(selectedGame && rPly < selectedGame.records.length ? selectedGame.records[rPly].pi : null);
  const rOptimal = $derived(rPly < rMoves.length ? analyze(rBoard).optimal : []);
  const rBits = $derived(
    selectedGame ? selectedGame.records.map((r) => -Math.log2(Math.max(r.pi[r.cell], 2 ** -10))) : []
  );

  const note = $derived.by(() => {
    const parts = [`roster ${roster.length}`];
    if (busy) parts.push(busy);
    else if (match) parts.push(`last duel: ${playerLabel(match.x)} ${match.xWins}-${match.draws}-${match.oWins} ${playerLabel(match.o)}`);
    if (selectedGame) parts.push(`replay ply ${rPly}/${rMoves.length}`);
    return parts.join(' · ');
  });

  const gameChip = (g: GameRecord) => (g.winner === 1 ? 'X' : g.winner === 2 ? 'O' : '=');
</script>

<TopBar {panels}>
  <div class="f">
    <span class="lbl">corpus</span>
    <div class="toggle-group">
      {#each KINDS as k (k)}
        <button class:active={kind === k} onclick={() => (kind = k)} title="Corpus the toy players train on">{k}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl">games</span>
    <div class="toggle-group">
      {#each N_OPTS as n (n)}
        <button class:active={nGames === n} onclick={() => (nGames = n)}>{n}</button>
      {/each}
    </div>
  </div>

  <div class="f">
    <span class="lbl" title="0 = argmax (deterministic); higher samples pi^(1/T)">temp</span>
    <div class="toggle-group">
      {#each TEMP_OPTS as t (t)}
        <button class:active={temp === t} onclick={() => (temp = t)}>{t}</button>
      {/each}
    </div>
  </div>

  <button class="ghost" title="Seeds toy training, the probe suite, and match sampling" onclick={() => (seed += 1)}>🎲 seed {seed}</button>

  {#if busy}<span class="mono busyflag">{busy}</span>{/if}
  <EngineStatus />
</TopBar>

{#snippet aRoster()}<span class="mono">{roster.length} players</span>{/snippet}
{#snippet pRoster()}
  <div class="pcol scrollbar">
    <div class="rlist">
      {#each roster as id (id)}
        <div class="rrow mono">
          <span class="rkind faint">{id.split(':')[0]}</span>
          <span class="rname">{playerLabel(id)}</span>
          <button class="ghost xbtn" disabled={roster.length <= 2} onclick={() => removeFromRoster(id)} title="Remove from roster">✕</button>
        </div>
      {/each}
    </div>

    <div class="adder">
      <span class="sub mono faint">add toy arm (trains on demand · {kind} · seed {seed})</span>
      <div class="arow2">
        <select bind:value={toyArch}>{#each ARCHS as a (a)}<option value={a}>{a}</option>{/each}</select>
        <select bind:value={toySignal}>{#each SIGNALS as s (s)}<option value={s} disabled={s === 'distill' && toyArch === 'gpt'}>{s}</option>{/each}</select>
        <button class="ghost" onclick={() => addToRoster(`toy:${toyArch}:${toySignal}`)}>+ add</button>
      </div>
    </div>

    <div class="adder">
      <span class="sub mono faint">add LLM player {engine.up ? '' : '(engine offline — make up)'}</span>
      <div class="arow2">
        <select bind:value={llmModel} disabled={!engine.up}>
          {#each engine.hub as m (m.name)}<option value={m.name}>{m.name}</option>{/each}
        </select>
        <select bind:value={llmMode} disabled={!engine.up}>
          <option value="auto">auto</option><option value="chat">chat</option><option value="raw">raw</option>
        </select>
        <button
          class="ghost"
          disabled={!engine.up}
          onclick={() => addToRoster(`llm:${llmModel}:${llmMode === 'auto' ? defaultLlmMode(llmModel) : llmMode}`)}
        >+ add</button>
      </div>
      <div class="faint hint">LLM moves are read as next-token distributions over the digit tokens — one /next forward per position (~50 ms), memoized.</div>
    </div>
  </div>
{/snippet}

{#snippet aMatch()}{#if match}<span class="mono">{match.xWins}-{match.draws}-{match.oWins}</span>{/if}{/snippet}
{#snippet pMatch()}
  <div class="pcol scrollbar">
    <div class="arow2">
      <label class="f"><span class="lbl">X</span>
        <select bind:value={xId}>{#each roster as id (id)}<option value={id}>{playerLabel(id)}</option>{/each}</select>
      </label>
      <label class="f"><span class="lbl">O</span>
        <select bind:value={oId}>{#each roster as id (id)}<option value={id}>{playerLabel(id)}</option>{/each}</select>
      </label>
      <button onclick={runMatch} disabled={!!busy}>▶ play {nGames}</button>
    </div>
    {#if errMsg}<div class="err mono">{errMsg}</div>{/if}
    {#if match}
      <div class="mono resline">
        {playerLabel(match.x)} <b>{match.xWins}</b> – {match.draws} – <b>{match.oWins}</b> {playerLabel(match.o)}
      </div>
      <div class="glist">
        {#each match.games as g, i (i)}
          <button
            class="gchip mono"
            class:sel={selectedGame === g}
            class:xwin={g.winner === 1}
            class:owin={g.winner === 2}
            onclick={() => pickGame(g)}
            title={`game ${i}: ${g.moves.join('')} — click to replay`}
          >{gameChip(g)} {g.moves.join('')}</button>
        {/each}
      </div>
      {#if temp === 0}<div class="faint hint">temperature 0: deterministic players repeat one game — raise temp for statistics</div>{/if}
    {/if}
  </div>
{/snippet}

{#snippet pTable()}
  <div class="pcol scrollbar">
    <button class="ghost" onclick={runTable} disabled={!!busy}>▶ run round-robin ({roster.length * (roster.length - 1)} matches × {nGames} games)</button>
    {#if tableResults.length}
      <MatchTable labels={roster.map((id) => [id, playerLabel(id)])} results={tableResults} onPick={(r) => { match = r; pickGame(r.games[0] ?? null); }} />
    {/if}
  </div>
{/snippet}

{#snippet aReplay()}{#if selectedGame}<span class="mono">{playerLabel(selectedGame.x)} vs {playerLabel(selectedGame.o)}</span>{/if}{/snippet}
{#snippet pReplay()}
  {#if selectedGame}
    <div class="pcol scrollbar">
      <TicBoard board={rBoard} pi={rPi} optimal={rOptimal} modelPick={rPly < rMoves.length ? rMoves[rPly] : null} />
      <div class="turnrow">
        <button class="ghost tbtn" disabled={rPly === 0} onclick={() => (replayPly = 0)}>⏮</button>
        <button class="ghost tbtn" disabled={rPly === 0} onclick={() => (replayPly = rPly - 1)}>◀</button>
        <input type="range" class="turnslider" min="0" max={rMoves.length} step="1" value={rPly}
          oninput={(e) => (replayPly = Number((e.currentTarget as HTMLInputElement).value))} />
        <button class="ghost tbtn" disabled={rPly >= rMoves.length} onclick={() => (replayPly = rPly + 1)}>▶</button>
        <button class="ghost tbtn" disabled={rPly >= rMoves.length} onclick={() => (replayPly = rMoves.length)}>⏭</button>
        <span class="mono faint plylabel">ply {rPly}/{rMoves.length}</span>
      </div>
      <div class="faint hint">board overlay = the MOVER's policy at this ply · yellow outline = the move actually played</div>
      <OutcomeStrip game={rMoves} bits={rBits} ply={rPly} onSeek={(k) => (replayPly = k)} />
      <a class="mono openlink" href={`#/tictac?g=${rMoves.join('')}`}>open in Tic·tac →</a>
    </div>
  {:else}
    <div class="panel empty"><span class="faint">play a duel, then click a game to replay it</span></div>
  {/if}
{/snippet}

{#snippet pReport()}
  <div class="pcol scrollbar">
    <button class="ghost" onclick={runReport} disabled={!!busy}>▶ run report cards ({roster.length} players × probe suite)</button>
    {#if cards.length}
      <table class="mono rtable">
        <thead><tr><th>player</th><th>agree</th><th>bits vs opt</th><th>equiv err</th><th>illegal</th><th>decisive</th></tr></thead>
        <tbody>
          {#each cards as { id, card } (id)}
            <tr>
              <td class="pname">{playerLabel(id)}</td>
              <td>{(card.agreement * 100).toFixed(0)}%</td>
              <td>{card.bitsVsOptimal.toFixed(2)}</td>
              <td>{card.equivariance.toFixed(3)}</td>
              <td>{(card.illegalMass * 100).toFixed(0)}%</td>
              <td>{id.startsWith('llm:') ? `${(card.decisiveness * 100).toFixed(0)}%` : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <div class="faint hint">agreement/bits/equivariance against the solved game over the {seed}-seeded probe suite · decisiveness = an LLM's next-token mass on ANY digit</div>
    {/if}
  </div>
{/snippet}

{#snippet pGuide()}
  <InterpretGuide lens="arena" sections={['arenaread']} />
{/snippet}

<PanelHost
  manager={panels}
  snippets={{ roster: pRoster, match: pMatch, table: pTable, replay: pReplay, report: pReport, guide: pGuide }}
  actions={{ roster: aRoster, match: aMatch, replay: aReplay }}
/>

<TransportBar {note} />

<style>
  .busyflag { font-size: 11px; color: var(--data); }
  .pcol { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .hint { font-size: 10.5px; line-height: 1.4; max-width: 340px; overflow-wrap: break-word; }
  .sub { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
  .err { color: var(--bad); font-size: 11px; }

  .rlist { display: flex; flex-direction: column; gap: 3px; }
  .rrow { display: flex; align-items: center; gap: 8px; font-size: 11.5px; padding: 2px 4px; border-radius: 4px; }
  .rrow:hover { background: var(--bg-2); }
  .rkind { flex: 0 0 44px; font-size: 9.5px; text-transform: uppercase; }
  .rname { flex: 1 1 auto; }
  .xbtn { padding: 0 6px; font-size: 10px; }
  .adder { display: flex; flex-direction: column; gap: 4px; padding-top: 6px; border-top: 1px solid var(--border-2); }
  .arow2 { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }

  .resline { font-size: 13px; }
  .resline b { color: var(--model); }
  .glist { display: flex; flex-wrap: wrap; gap: 4px; }
  .gchip {
    font-size: 10px; padding: 2px 6px;
    border: 1px solid var(--border-2); border-radius: 4px;
    background: var(--bg-2); color: var(--muted); cursor: pointer;
  }
  .gchip.xwin { border-color: var(--model); color: var(--model); }
  .gchip.owin { border-color: var(--data); color: var(--data); }
  .gchip.sel { outline: 1.5px solid var(--chosen); }

  .turnrow { display: flex; align-items: center; gap: 4px; max-width: 300px; }
  .tbtn { padding: 2px 7px; font-size: 11px; }
  .turnslider { flex: 1 1 auto; min-width: 60px; }
  .plylabel { font-size: 10.5px; white-space: nowrap; }
  .openlink { font-size: 11px; color: var(--model); }

  .rtable { border-collapse: collapse; font-size: 11px; }
  .rtable th, .rtable td { padding: 4px 8px; border: 1px solid var(--border-2); text-align: right; }
  .rtable th { color: var(--muted); font-weight: 500; }
  .rtable .pname { text-align: left; }
</style>
