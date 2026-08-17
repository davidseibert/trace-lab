<script lang="ts">
  import { crossEntropyBits, entropyBits, informationBits, klDivergenceBits } from '../../../lib/math/derive';

  let mode = $state<'soft' | 'onehot'>('soft');
  let qA = $state(0.8);
  let pA = $state(0.55);
  const q = $derived(mode === 'onehot' ? [1, 0] : [qA, 1 - qA]);
  const p = $derived([pA, 1 - pA]);
  const h = $derived(entropyBits(q));
  const ce = $derived(crossEntropyBits(q, p));
  const kl = $derived(klDivergenceBits(q, p));
  const gradA = $derived(p[0] - q[0]);
  const gradB = $derived(p[1] - q[1]);
</script>

<div class="lab scrollbar">
  <div class="head">
    <div class="toggle-group">
      <button class:active={mode === 'soft'} onclick={() => (mode = 'soft')}>soft target q</button>
      <button class:active={mode === 'onehot'} onclick={() => (mode = 'onehot')}>observed A</button>
    </div>
    <span class="big mono">H(q,p) = {ce.toFixed(3)} bits</span>
  </div>

  <div class="sliders">
    {#if mode === 'soft'}
      <label><span>target q(A)</span><input type="range" min="0.02" max="0.98" step="0.01" bind:value={qA} /><b class="mono">{qA.toFixed(2)}</b></label>
    {/if}
    <label><span>model p(A)</span><input type="range" min="0.02" max="0.98" step="0.01" bind:value={pA} /><b class="mono">{pA.toFixed(2)}</b></label>
  </div>

  <div class="distributions">
    <div><span class="name">target q</span><div class="bar"><i style={`width:${q[0] * 100}%`}></i><em></em></div><span class="mono">A {q[0].toFixed(2)} · B {q[1].toFixed(2)}</span></div>
    <div><span class="name">model p</span><div class="bar model"><i style={`width:${p[0] * 100}%`}></i><em></em></div><span class="mono">A {p[0].toFixed(2)} · B {p[1].toFixed(2)}</span></div>
  </div>

  <div class="decomp">
    <div class="segment entropy" style={`flex:${Math.max(h, .001)}`}><span>H(q)</span><b class="mono">{h.toFixed(3)}</b></div>
    <div class="segment kl" style={`flex:${Math.max(kl, .001)}`}><span>KL(q‖p)</span><b class="mono">{kl.toFixed(3)}</b></div>
  </div>
  <div class="identity mono">H(q,p) = H(q) + KL(q‖p) = {h.toFixed(3)} + {kl.toFixed(3)} = {ce.toFixed(3)} bits</div>

  <div class="terms mono">
    <div><span>A term</span><b>{q[0].toFixed(2)} × {informationBits(p[0]).toFixed(3)} = {(q[0] * informationBits(p[0])).toFixed(3)}</b></div>
    <div><span>B term</span><b>{q[1].toFixed(2)} × {informationBits(p[1]).toFixed(3)} = {(q[1] * informationBits(p[1])).toFixed(3)}</b></div>
  </div>

  <div class="gradient">
    <div><span class="faint">gradient on A logit</span><b class="mono" class:down={gradA > 0}>pA − qA = {gradA.toFixed(3)}</b></div>
    <div><span class="faint">gradient on B logit</span><b class="mono" class:down={gradB > 0}>pB − qB = {gradB.toFixed(3)}</b></div>
    <p>Gradient descent subtracts these values: underpredicted outcomes move up; overpredicted outcomes move down.</p>
  </div>
</div>

<style>
  .lab { display: flex; flex-direction: column; gap: 13px; overflow: auto; min-height: 0; }
  .head { display: flex; justify-content: space-between; gap: 10px; align-items: center; flex-wrap: wrap; }
  .big { color: var(--total); font-size: 18px; font-weight: 700; }
  .sliders { display: flex; flex-direction: column; gap: 6px; }
  .sliders label { display: grid; grid-template-columns: 90px 1fr 42px; gap: 8px; align-items: center; font-size: 11px; color: var(--muted); }
  input[type='range'] { width: 100%; accent-color: var(--model); }
  .distributions { display: flex; flex-direction: column; gap: 7px; }
  .distributions > div { display: grid; grid-template-columns: 65px 1fr 110px; gap: 8px; align-items: center; font-size: 10.5px; }
  .name { color: var(--muted); }
  .bar { height: 18px; display: flex; border: 1px solid var(--border); border-radius: var(--r-sm); overflow: hidden; background: var(--bg-2); }
  .bar i { background: var(--data); }
  .bar em { flex: 1; background: rgba(255,180,84,.25); }
  .bar.model i { background: var(--model); }
  .bar.model em { background: rgba(91,156,255,.25); }
  .decomp { min-height: 48px; display: flex; border-radius: var(--r-sm); overflow: hidden; }
  .segment { min-width: 3px; padding: 7px 9px; display: flex; justify-content: space-between; align-items: center; gap: 8px; transition: flex .15s; }
  .segment span { white-space: nowrap; font-size: 10px; }
  .segment.entropy { background: var(--data); color: #241300; }
  .segment.kl { background: var(--model); color: #06121f; }
  .identity { text-align: center; color: var(--muted); font-size: 10.5px; }
  .terms { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .terms div { padding: 7px 9px; display: flex; justify-content: space-between; gap: 8px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm); font-size: 10.5px; }
  .terms span { color: var(--faint); }
  .gradient { border-top: 1px solid var(--border); padding-top: 9px; display: grid; grid-template-columns: 1fr 1fr; gap: 7px 12px; }
  .gradient div { display: flex; flex-direction: column; }
  .gradient b { color: var(--good); }
  .gradient b.down { color: var(--bad); }
  .gradient p { grid-column: 1 / -1; margin: 0; color: var(--muted); font-size: 11px; }
</style>
