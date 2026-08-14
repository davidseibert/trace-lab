<script lang="ts">
  import type { TensorSnap } from '../../lib/llm/tensor';

  let {
    attn,
    tokens,
    tokenColors,
    focusPos,
    onCellClick
  }: {
    attn: TensorSnap; // [H, T, T]
    tokens: string[];
    tokenColors: string[];
    /** Query position the arc diagram fans out from. Defaults to the last token. */
    focusPos?: number;
    /** When supplied, per-head heatmap cells become clickable. */
    onCellClick?: (head: number, row: number, col: number, value: number) => void;
  } = $props();

  const H = $derived(attn.shape[0]);
  const T = $derived(attn.shape[1]);
  const heads = $derived([...Array(H).keys()]);
  const idx = $derived([...Array(T).keys()]);

  const at = (h: number, i: number, j: number) => attn.data[h * T * T + i * T + j];

  // Arc diagram: how the focused token distributes its attention over
  // earlier tokens, averaged across heads. Defaults to the last token (the
  // one whose next-token we predict).
  const lastRow = $derived.by(() => {
    const q = focusPos ?? T - 1;
    const w: number[] = [];
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let h = 0; h < H; h++) s += at(h, q, j);
      w.push(s / H);
    }
    return w;
  });

  // Arc geometry in a 0..100 × 0..40 viewBox.
  const VW = 100;
  const VH = 42;
  const xAt = (i: number) => (T <= 1 ? VW / 2 : 6 + (i / (T - 1)) * (VW - 12));
  const baseY = VH - 8;
  const focusIdx = $derived(focusPos ?? T - 1);
  function arc(i: number): string {
    const x0 = xAt(focusIdx);
    const x1 = xAt(i);
    const peak = baseY - 6 - Math.abs(x0 - x1) * 0.4;
    const mid = (x0 + x1) / 2;
    return `M ${x0} ${baseY} Q ${mid} ${peak} ${x1} ${baseY}`;
  }
</script>

<div class="attn">
  <!-- Arc diagram for the focused token -->
  <svg class="arcs" viewBox="0 0 {VW} {VH}" preserveAspectRatio="none" role="presentation">
    {#each lastRow as w, j}
      {#if j !== focusIdx && w > 0.01}
        <path d={arc(j)} class="arc" style="stroke:{tokenColors[j]}; stroke-opacity:{Math.min(1, 0.15 + w)}; stroke-width:{(0.4 + w * 3).toFixed(2)}" />
      {/if}
    {/each}
    {#each tokens as _, j}
      <circle cx={xAt(j)} cy={baseY} r="1.6" class="node" class:q={j === focusIdx} />
    {/each}
  </svg>
  <div class="arc-labels">
    {#each tokens as t, j}
      <span class="alab mono" class:q={j === focusIdx} style="color:{tokenColors[j]}">{t}</span>
    {/each}
  </div>

  <!-- Per-head heatmaps -->
  <div class="heads scrollbar">
    {#each heads as h}
      <div class="head">
        <div class="head-title faint mono">head {h}</div>
        <div class="hmap" style="grid-template-columns:repeat({T}, 1fr)">
          {#each idx as i}
            {#each idx as j}
              {@const w = at(h, i, j)}
              {@const clickable = !!onCellClick && j <= i}
              <div
                class="hcell"
                class:masked={j > i}
                class:clickable
                style="background:rgba(91,156,255,{j > i ? 0 : (0.05 + 0.95 * w).toFixed(3)})"
                title={`${tokens[i]} → ${tokens[j]} : ${(w * 100).toFixed(0)}%`}
                role={clickable ? 'button' : undefined}
                tabindex={clickable ? 0 : undefined}
                onclick={() => clickable && onCellClick?.(h, i, j, w)}
                onkeydown={(e) => {
                  if (clickable && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    onCellClick?.(h, i, j, w);
                  }
                }}
              ></div>
            {/each}
          {/each}
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  .attn { display: flex; flex-direction: column; gap: 6px; min-height: 0; flex: 1 1 auto; }
  .arcs { width: 100%; height: 56px; flex: 0 0 auto; }
  .arc { fill: none; stroke-linecap: round; }
  .node { fill: var(--faint); }
  .node.q { fill: var(--chosen); r: 2.2; }
  .arc-labels { display: flex; justify-content: space-between; gap: 2px; flex: 0 0 auto; padding: 0 4px; }
  .alab { font-size: 10px; opacity: 0.7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .alab.q { opacity: 1; font-weight: 700; }

  .heads { display: flex; gap: 12px; overflow: auto; min-height: 0; flex: 1 1 auto; align-items: flex-start; }
  .head { flex: 1 1 0; min-width: 90px; display: flex; flex-direction: column; gap: 4px; }
  .head-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
  .hmap { display: grid; gap: 1px; aspect-ratio: 1 / 1; background: var(--bg-2); padding: 2px; border-radius: 4px; }
  .hcell { border-radius: 1px; min-height: 0; }
  .hcell.masked { background: transparent !important; }
  .hcell.clickable:not(.masked) { cursor: pointer; }
  .hcell.clickable:not(.masked):hover { outline: 1.5px solid var(--text); outline-offset: -1.5px; }
</style>
