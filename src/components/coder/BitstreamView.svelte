<script lang="ts">
  import { shortestCodeword, type CoderStep, type Codeword } from '../../lib/coder/arithmetic';
  import { symColorFor } from '../../lib/coder/colors';

  let {
    step,
    codeword,
    idealTotal,
    final
  }: {
    step: CoderStep;
    codeword: Codeword;
    idealTotal: number;
    /** True at the last step — the so-far codeword has become the final one. */
    final: boolean;
  } = $props();

  // Colour the emitted symbols by matching them back to the step's distribution.
  const idForLabel = $derived(new Map(step.dist.map((e) => [e.label, e.id])));

  const isEncode = $derived(step.phase === 'encode');
  // The shortest codeword that names the interval AS IT STANDS — grows while
  // scrubbing, and equals the final codeword only at the end.
  const live = $derived(shortestCodeword(step.newLo, step.newHi));
</script>

<div class="bs">
  <div class="headline">
    <div class="big">
      <span class="num mono">{step.bitsSoFar.toFixed(2)}</span>
      <span class="unit">bits {isEncode ? 'so far' : 'resolved'}</span>
    </div>
    <span class="muted">ideal = Σ −log₂p</span>
  </div>

  <div class="emitted">
    {#each step.emitted as lab, i (i)}
      {@const id = idForLabel.get(lab)}
      <span class="ec mono" style={id !== undefined ? `color:${symColorFor(id).fg}` : ''}>{lab}</span>
    {/each}
  </div>

  <div class="checkout">
    {#if isEncode}
      <div class="crow">
        <span class="clbl">{final ? 'codeword' : 'codeword so far'}</span>
        <span class="cval mono code">0.{live.bits}</span>
      </div>
      <div class="crow">
        <span class="clbl">that is</span>
        <span class="cval mono">{live.nbits} bits</span>
      </div>
      <div class="crow">
        <span class="clbl">ideal total (whole string)</span>
        <span class="cval mono">{idealTotal.toFixed(2)} bits</span>
      </div>
      <p class="foot muted">
        The leading bits aren't final until the interval is — a later symbol can
        carry and flip them. (That's why a real streaming coder buffers /
        renormalises instead of emitting bits as it goes.)
      </p>
    {:else}
      <div class="crow">
        <span class="clbl">codeword (input)</span>
        <span class="cval mono code">0.{codeword.bits}</span>
      </div>
      <div class="crow">
        <span class="clbl">read as</span>
        <span class="cval mono">{codeword.value.toFixed(5)}</span>
      </div>
      <p class="foot muted">
        One number, decoded top-down: each symbol is the slice it lands in, then
        we zoom into that slice and read the next.
      </p>
    {/if}
  </div>
</div>

<style>
  .bs { display: flex; flex-direction: column; gap: 10px; flex: 1 1 auto; min-height: 0; }
  .headline { display: flex; align-items: baseline; justify-content: space-between; }
  .big { display: flex; align-items: baseline; gap: 6px; }
  .num { font-size: 24px; font-weight: 700; color: var(--data); }
  .unit { font-size: 11px; color: var(--muted); }

  .emitted { display: flex; flex-wrap: wrap; gap: 2px; font-size: 15px; padding: 6px 8px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm); min-height: 30px; }
  .ec { letter-spacing: 0.02em; }

  .checkout { display: flex; flex-direction: column; gap: 4px; }
  .crow { display: flex; justify-content: space-between; align-items: baseline; font-size: 12.5px; }
  .clbl { color: var(--muted); }
  .cval.code { color: var(--total); word-break: break-all; text-align: right; }
  .foot { margin: 4px 0 0; font-size: 11px; line-height: 1.4; }
</style>
