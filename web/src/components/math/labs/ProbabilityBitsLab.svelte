<script lang="ts">
  import { informationBits } from '../../../lib/math/derive';

  let bits = $state(3);
  const p = $derived(2 ** -bits);
  const alternatives = $derived(1 / p);
  const measured = $derived(informationBits(p));
</script>

<div class="lab scrollbar">
  <div class="lab-head">
    <div>
      <div class="big mono">{measured.toFixed(2)} <span>bits</span></div>
      <div class="faint">information in an event with probability {(p * 100).toFixed(p < 0.01 ? 3 : 1)}%</div>
    </div>
    <div class="landmarks">
      {#each [0, 1, 2, 3, 4, 6, 8] as b}
        <button class:active={bits === b} onclick={() => (bits = b)}>{b}b</button>
      {/each}
    </div>
  </div>

  <label class="control">
    <span class="lbl">surprisal</span>
    <input type="range" min="0" max="12" step="0.1" bind:value={bits} />
    <span class="mono">{bits.toFixed(1)} b</span>
  </label>

  <div class="chain mono">
    <div><span>p</span><b>{p.toPrecision(4)}</b></div>
    <span class="arrow">→</span>
    <div><span>1 / p</span><b>{alternatives.toFixed(alternatives < 100 ? 1 : 0)} alternatives</b></div>
    <span class="arrow">→</span>
    <div><span>log₂(1 / p)</span><b>{measured.toFixed(2)} bits</b></div>
  </div>

  <div class="capacity">
    <div class="cap-head"><span>one event's share of probability space</span><span class="mono">p = 2<sup>−{bits.toFixed(1)}</sup></span></div>
    <div class="track"><div class="fill" style={`width:${Math.max(0.25, p * 100)}%`}></div></div>
  </div>

  <div class="proof">
    <span class="mono">I(pq) = I(p) + I(q)</span>
    <p>When two probabilities multiply, their bit costs add. That is the property logarithms supply.</p>
  </div>
</div>

<style>
  .lab { display: flex; flex-direction: column; gap: 16px; overflow: auto; min-height: 0; }
  .lab-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
  .big { color: var(--data); font-size: 28px; font-weight: 700; }
  .big span { font-size: 12px; color: var(--muted); }
  .landmarks { display: flex; flex-wrap: wrap; gap: 3px; justify-content: flex-end; }
  .landmarks button { padding: 3px 6px; font-size: 10px; }
  .landmarks button.active { background: var(--model); color: #06121f; border-color: var(--model); }
  .control { display: grid; grid-template-columns: auto 1fr 60px; align-items: center; gap: 8px; }
  input[type='range'] { width: 100%; accent-color: var(--model); }
  .chain { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: center; gap: 8px; }
  .chain div { min-width: 0; padding: 9px; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--bg-2); display: flex; flex-direction: column; gap: 3px; }
  .chain div span { color: var(--faint); font-size: 10px; }
  .chain div b { color: var(--text); font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; }
  .arrow { color: var(--faint); }
  .capacity { display: flex; flex-direction: column; gap: 5px; }
  .cap-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 10.5px; }
  .track { height: 24px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm); overflow: hidden; }
  .fill { height: 100%; background: var(--data); min-width: 2px; transition: width .15s; }
  .proof { border-top: 1px solid var(--border); padding-top: 10px; }
  .proof .mono { color: var(--total); }
  .proof p { margin: 4px 0 0; color: var(--muted); font-size: 11.5px; }
  @media (max-width: 560px) { .chain { grid-template-columns: 1fr; } .arrow { transform: rotate(90deg); text-align: center; } }
</style>
