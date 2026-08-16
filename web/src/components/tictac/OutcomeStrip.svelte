<script lang="ts">
  /**
   * The outcome poset, drawn: per ply of the current game, which of the three
   * outcomes (X win / draw / O win) is still achievable under ANY legal
   * continuation. The set only ever shrinks — that monotone map is the
   * game DAG's Hasse-diagram structure made visible. Below each played move,
   * the bits the model paid for it at the current training step.
   */
  import { analyze, boardFromMoves, DRAW, O_WIN, X_WIN } from '../../lib/tictac/game';

  let {
    game,
    bits
  }: {
    game: number[];
    /** bits[k] = −log₂ π(move k) at the current step (length = game.length). */
    bits: number[];
  } = $props();

  const plies = $derived.by(() =>
    [...Array(game.length + 1).keys()].map((k) => ({
      k,
      achievable: analyze(boardFromMoves(game.slice(0, k))).achievable,
      move: k < game.length ? game[k] : null,
      cost: k < game.length ? bits[k] : null
    }))
  );

  const MAX_BITS = 6;
</script>

<div class="strip scrollbar">
  {#each plies as p (p.k)}
    <div class="ply">
      <div class="chips">
        <span class="chip mono" class:on={(p.achievable & X_WIN) !== 0} style="--c: var(--model)" title="X win still achievable?">✕</span>
        <span class="chip mono" class:on={(p.achievable & DRAW) !== 0} style="--c: var(--muted)" title="draw still achievable?">=</span>
        <span class="chip mono" class:on={(p.achievable & O_WIN) !== 0} style="--c: var(--data)" title="O win still achievable?">◯</span>
      </div>
      {#if p.move !== null}
        <div class="move mono faint">→{p.move}</div>
        <div class="btrack" title={`model paid ${p.cost!.toFixed(2)} bits for move ${p.move} at this step`}>
          <div class="bfill" style="height:{Math.min(100, (p.cost! / MAX_BITS) * 100)}%"></div>
        </div>
        <div class="bval mono faint">{p.cost!.toFixed(1)}b</div>
      {:else}
        <div class="move mono faint">·</div>
      {/if}
    </div>
  {/each}
</div>
{#if game.length === 0}
  <div class="faint hint">Play moves on the board — outcomes disappear as the game descends its DAG, and each move shows what the model would have paid for it.</div>
{/if}

<style>
  .strip { display: flex; gap: 8px; overflow-x: auto; align-items: flex-start; padding-bottom: 4px; }
  .ply { display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 0 0 auto; }
  .chips { display: flex; flex-direction: column; gap: 2px; }
  .chip {
    font-size: 10px; width: 20px; height: 16px;
    display: inline-flex; align-items: center; justify-content: center;
    border: 1px solid var(--border-2); border-radius: 3px;
    color: var(--border-2); background: var(--bg-2); opacity: 0.35;
  }
  .chip.on { color: var(--c); border-color: var(--c); opacity: 1; }
  .move { font-size: 10px; }
  .btrack { width: 8px; height: 34px; background: var(--bg-2); border-radius: 3px; display: flex; flex-direction: column-reverse; overflow: hidden; }
  .bfill { background: var(--data); border-radius: 3px; }
  .bval { font-size: 9px; }
  .hint { font-size: 10.5px; margin-top: 4px; }
</style>
