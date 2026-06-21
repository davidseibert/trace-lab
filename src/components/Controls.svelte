<script lang="ts">
  import type { Player } from '../lib/player.svelte';

  let { player }: { player: Player<unknown, unknown> } = $props();

  const speeds = [0.5, 1, 1.5, 3, 6];
</script>

<div class="controls">
  <div class="transport">
    <button class="ghost" onclick={() => player.reset()} disabled={player.atStart} title="Reset to step 0">⏮</button>
    <button class="ghost" onclick={() => player.stepBack()} disabled={player.atStart} title="Step back">◀</button>
    <button class="primary play" onclick={() => player.toggle()} title="Play / pause">
      {player.playing ? '❚❚' : '▶'}
    </button>
    <button class="ghost" onclick={() => player.stepForward()} disabled={player.atEnd} title="Step forward">▶</button>
  </div>

  <div class="scrub">
    <input
      type="range"
      min="0"
      max={Math.max(0, player.count - 1)}
      value={player.index}
      oninput={(e) => player.seek(+(e.currentTarget as HTMLInputElement).value)}
    />
    <span class="counter mono">step {player.index} / {Math.max(0, player.count - 1)}</span>
  </div>

  <div class="speed">
    <span class="muted">speed</span>
    <div class="toggle-group">
      {#each speeds as s}
        <button class:active={player.speed === s} onclick={() => player.setSpeed(s)}>{s}×</button>
      {/each}
    </div>
  </div>
</div>

<style>
  .controls { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .transport { display: flex; gap: 4px; }
  .play { min-width: 44px; font-size: 13px; }
  .scrub { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 220px; }
  input[type='range'] { flex: 1; accent-color: var(--model); }
  .counter { font-size: 12.5px; color: var(--muted); white-space: nowrap; }
  .speed { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
</style>
