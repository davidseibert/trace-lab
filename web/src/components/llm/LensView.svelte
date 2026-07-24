<script lang="ts">
  import type { LensReport } from '../../lib/llm/lens';
  import { bitsOf } from '../../lib/llm/lens';

  let {
    report,
    vocab,
    colors,
    focusId
  }: {
    report: LensReport;
    vocab: string[];
    colors: string[];
    /** Token whose code length the bits ladder tracks (the active prediction). */
    focusId: number;
  } = $props();

  let mode = $state<'naive' | 'j'>('naive');

  const rows = $derived(
    report.rungs.map((r) => {
      const probs = mode === 'j' ? r.jProbs : r.naiveProbs;
      const top = vocab
        .map((tok, id) => ({ tok, id, p: probs[id] }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);
      return { rung: r, probs, top, bits: bitsOf(probs, focusId) };
    })
  );

  // Bar scale: the uniform prior is the "knows nothing" reference; codes can
  // exceed it (worse than uniform), so leave headroom up to 2× uniform.
  const maxBits = $derived(report.uniformBits * 2);
  const barW = (bits: number) => `${Math.min(100, (bits / maxBits) * 100)}%`;
</script>

<div class="lens scrollbar">
  <div class="modes">
    <button class="mode" class:on={mode === 'naive'} onclick={() => (mode = 'naive')}
      title="Decode each rung's residual straight through the final LayerNorm + unembed — as if the network were done.">
      logit lens
    </button>
    <button class="mode" class:on={mode === 'j'} onclick={() => (mode = 'j')}
      title="Transport each rung through the exact Jacobian of the remaining layers first, then decode — the faithful readout.">
      J-lens
    </button>
    <span class="mono faint uni" title="log₂(V): the cost of the next token before the model knows anything">
      uniform {report.uniformBits.toFixed(1)}b
    </span>
  </div>

  {#each rows as { rung, top, bits } (rung.key)}
    <div class="rung">
      <div class="rhead">
        <span class="rlabel mono">{rung.label}</span>
        <span class="vis mono" title="share of this residual's length the output can see at all — the rest is blind directions (0 bits about the next token)">
          <span class="vis-track"><span class="vis-fill" style="width:{(rung.visibleFrac * 100).toFixed(0)}%"></span></span>
          {(rung.visibleFrac * 100).toFixed(0)}% visible
        </span>
      </div>
      <div class="rbody">
        <div class="tops">
          {#each top as { tok, id, p } (id)}
            <span class="tk mono" class:focus={id === focusId} style="color:{colors[id]}; border-color:{colors[id]}">
              {tok} <span class="pct">{(p * 100).toFixed(0)}%</span>
            </span>
          {/each}
        </div>
        <div class="bitsbar" title={`−log₂ p(${vocab[focusId]}) at this rung`}>
          <span class="bb-track">
            <span class="bb-fill" style="width:{barW(bits)}"></span>
            <span class="bb-uniform" style="left:{barW(report.uniformBits)}"></span>
          </span>
          <span class="bb-num mono">{bits.toFixed(2)}b</span>
        </div>
      </div>
    </div>
  {/each}

  <div class="ladder mono faint">
    −log₂ p(<span style="color:{colors[focusId]}">{vocab[focusId]}</span>):
    {rows.map((r) => r.bits.toFixed(1)).join(' → ')} bits — the code
    {rows.length && rows[rows.length - 1].bits <= rows[0].bits ? 'shortens' : 'lengthens'} with depth
  </div>
</div>

<style>
  .lens { display: flex; flex-direction: column; gap: 7px; overflow: auto; min-height: 0; flex: 1 1 auto; }

  .modes { display: flex; align-items: center; gap: 6px; }
  .mode {
    font-size: 10.5px; padding: 2px 8px; border-radius: 4px;
    border: 1px solid var(--border-2); background: var(--bg-2);
    color: var(--muted); cursor: pointer;
  }
  .mode.on { color: var(--text); border-color: var(--model); background: rgba(91, 156, 255, 0.12); }
  .uni { margin-left: auto; font-size: 10px; }

  .rung { display: flex; flex-direction: column; gap: 3px; padding: 5px 6px; border: 1px solid var(--border-2); border-radius: 5px; background: var(--bg-2); }
  .rhead { display: flex; align-items: center; gap: 8px; }
  .rlabel { font-size: 11px; color: var(--text); }
  .vis { margin-left: auto; display: flex; align-items: center; gap: 5px; font-size: 9.5px; color: var(--muted); }
  .vis-track { width: 42px; height: 5px; border-radius: 3px; background: var(--bg); overflow: hidden; display: inline-block; }
  .vis-fill { display: block; height: 100%; background: var(--model); border-radius: 3px; }

  .rbody { display: flex; align-items: center; gap: 8px; }
  .tops { display: flex; gap: 4px; flex: 1 1 auto; min-width: 0; flex-wrap: wrap; }
  .tk {
    font-size: 10.5px; padding: 1px 6px; border: 1px solid; border-radius: 4px;
    background: var(--bg); white-space: nowrap;
  }
  .tk .pct { color: var(--muted); font-size: 9.5px; }
  .tk.focus { outline: 1px solid rgba(255, 209, 102, 0.5); }

  .bitsbar { display: flex; align-items: center; gap: 6px; flex: 0 0 auto; width: 118px; }
  .bb-track { position: relative; flex: 1; height: 8px; border-radius: 3px; background: var(--bg); overflow: hidden; }
  .bb-fill { display: block; height: 100%; background: var(--data); border-radius: 3px; transition: width 0.15s ease; }
  .bb-uniform { position: absolute; top: -1px; bottom: -1px; width: 1px; background: var(--muted); opacity: 0.7; }
  .bb-num { font-size: 10px; color: var(--data); min-width: 40px; text-align: right; }

  .ladder { font-size: 10.5px; line-height: 1.4; }
</style>
