<script lang="ts">
  import { Player } from '../../../lib/player.svelte';
  import { binaryStream } from '../../../lib/math/derive';
  import { encode, idealBits, type CoderStep } from '../../../lib/coder/arithmetic';
  import Controls from '../../Controls.svelte';
  import IntervalView from '../../coder/IntervalView.svelte';

  let pA = $state(0.6);
  let message = $state('ABBA');
  const stream = $derived(binaryStream(message, pA));
  const encoded = $derived(encode(stream));
  const ideal = $derived(idealBits(encoded.steps));
  const player = new Player<CoderStep>();

  $effect(() => player.reload(encoded.steps));
  const cur = $derived(player.current);
</script>

<div class="lab">
  <div class="controls-row">
    <label class="prob"><span class="lbl">p(A)</span><input type="range" min="0.1" max="0.9" step="0.05" bind:value={pA} /><span class="mono">{pA.toFixed(2)}</span></label>
    <div class="messages">
      {#each ['AAAA', 'ABBA', 'BBBB'] as m}<button class:active={message === m} onclick={() => (message = m)}>{m}</button>{/each}
    </div>
  </div>

  <div class="receipt mono">
    <span>message {message}</span>
    <span>ideal {ideal.toFixed(3)} bits</span>
    <span>literal 0.{encoded.codeword.bits}₂ ({encoded.codeword.nbits} bits)</span>
  </div>

  {#if cur}
    <div class="invariant mono">
      <span>step {player.index + 1}</span>
      <b>new width = old width × {cur.chosenP.toFixed(3)}</b>
      <span>{(cur.hi - cur.lo).toPrecision(4)} × {cur.chosenP.toFixed(3)} = {(cur.newHi - cur.newLo).toPrecision(4)}</span>
    </div>
    <div class="interval-wrap"><IntervalView steps={player.steps} index={player.index} memoryless /></div>
    <div class="transport"><Controls {player} /><span class="mono faint">−log₂(width) = {cur.bitsSoFar.toFixed(3)} accumulated bits</span></div>
  {/if}
</div>

<style>
  .lab { display: flex; flex-direction: column; gap: 10px; min-height: 0; height: 100%; }
  .controls-row { display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
  .prob { display: grid; grid-template-columns: auto 120px 40px; gap: 7px; align-items: center; }
  input[type='range'] { width: 100%; accent-color: var(--model); }
  .messages { display: flex; gap: 3px; }
  .messages button { padding: 3px 7px; font-size: 10px; }
  .messages button.active { background: var(--model); color: #06121f; border-color: var(--model); }
  .receipt { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 12px; padding: 7px 9px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm); color: var(--muted); font-size: 10.5px; }
  .receipt span:nth-child(2) { color: var(--data); }
  .receipt span:last-child { color: var(--total); }
  .invariant { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; font-size: 10.5px; }
  .invariant b { color: var(--chosen); text-align: center; }
  .invariant span { color: var(--muted); }
  .interval-wrap { flex: 1 1 auto; min-height: 150px; display: flex; }
  .transport { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid var(--border); padding-top: 7px; }
</style>
