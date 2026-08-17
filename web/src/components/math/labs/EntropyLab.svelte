<script lang="ts">
  import { effectiveChoices, entropyBits, informationBits, normalize } from '../../../lib/math/derive';

  let weights = $state([6, 3, 1]);
  const p = $derived(normalize(weights));
  const terms = $derived(p.map((q) => q * informationBits(q)));
  const h = $derived(entropyBits(p));
  const maxH = Math.log2(3);
  const colors = ['var(--model)', 'var(--data)', 'var(--chosen)'];

  function setWeight(i: number, value: number) {
    weights = weights.map((w, j) => (i === j ? value : w));
  }
</script>

<div class="lab scrollbar">
  <div class="headline">
    <div><span class="big mono">H = {h.toFixed(3)} bits</span><span class="faint"> per draw</span></div>
    <div class="effective mono">2<sup>H</sup> = {effectiveChoices(h).toFixed(2)} effective choices</div>
  </div>

  <div class="mix" title="probability distribution">
    {#each p as q, i}<div style={`flex:${q};background:${colors[i]}`} title={`p${i + 1}=${q.toFixed(3)}`}></div>{/each}
  </div>

  <div class="controls">
    {#each weights as w, i}
      <label>
        <span class="swatch" style={`background:${colors[i]}`}></span>
        <span class="mono">p{i + 1} = {p[i].toFixed(3)}</span>
        <input type="range" min="0" max="10" step="0.1" value={w} oninput={(e) => setWeight(i, +(e.currentTarget as HTMLInputElement).value)} />
      </label>
    {/each}
  </div>

  <div class="terms">
    <div class="thead mono faint"><span>outcome</span><span>surprisal</span><span>weighted term</span></div>
    {#each p as q, i}
      <div class="row mono">
        <span>p{i + 1} = {q.toFixed(3)}</span>
        <span>−log₂p = {informationBits(q).toFixed(3)}</span>
        <span style={`color:${colors[i]}`}>{q.toFixed(3)} × {informationBits(q).toFixed(3)} = {terms[i].toFixed(3)}</span>
      </div>
    {/each}
    <div class="sum mono"><span>sum</span><b>{terms.map((x) => x.toFixed(3)).join(' + ')} = {h.toFixed(3)} bits</b></div>
  </div>

  <div class="meter">
    <div class="mhead"><span>certainty</span><span>current H</span><span>uniform maximum log₂3 = {maxH.toFixed(3)}</span></div>
    <div class="track"><div class="fill" style={`width:${(h / maxH) * 100}%`}></div></div>
  </div>
</div>

<style>
  .lab { display: flex; flex-direction: column; gap: 14px; overflow: auto; min-height: 0; }
  .headline { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .big { color: var(--total); font-size: 21px; font-weight: 700; }
  .effective { color: var(--data); font-size: 11.5px; }
  .mix { height: 28px; display: flex; overflow: hidden; border: 1px solid var(--border); border-radius: var(--r-sm); }
  .mix div { min-width: 1px; transition: flex .15s; }
  .controls { display: flex; flex-direction: column; gap: 6px; }
  .controls label { display: grid; grid-template-columns: 12px 92px 1fr; align-items: center; gap: 8px; }
  input[type='range'] { width: 100%; accent-color: var(--model); }
  .terms { border: 1px solid var(--border); border-radius: var(--r-sm); overflow: hidden; }
  .thead, .row { display: grid; grid-template-columns: .8fr 1fr 1.5fr; gap: 8px; padding: 5px 8px; font-size: 10.5px; }
  .thead { background: var(--panel-2); text-transform: uppercase; letter-spacing: .04em; }
  .row + .row { border-top: 1px solid var(--border); }
  .sum { display: flex; justify-content: space-between; gap: 8px; padding: 7px 8px; border-top: 1px solid var(--border-2); color: var(--muted); font-size: 10.5px; }
  .sum b { color: var(--total); text-align: right; }
  .meter { display: flex; flex-direction: column; gap: 4px; }
  .mhead { display: flex; justify-content: space-between; color: var(--faint); font-size: 9.5px; }
  .track { height: 12px; border-radius: 6px; background: var(--bg-2); overflow: hidden; }
  .fill { height: 100%; background: var(--total); transition: width .15s; }
</style>
