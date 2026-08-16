<script lang="ts">
  /**
   * The circuits-repo readout, transplanted: (1) a weight-magnitude threshold
   * slider — L1 during training, threshold only at inspection — with the count
   * of surviving FFN edges and live hidden units; (2) each hidden unit's
   * Pearson correlation against the game's KNOWN feature basis (line threats,
   * side to move, cell occupancy); (3) the token embeddings' top-2 PCs, where
   * D₄ orbit structure (corners / edges / center) should appear geometrically.
   */
  import type { TensorSnap } from '../../lib/llm/tensor';
  import type { SparsityReport } from '../../lib/tictac/ticTrain';
  import { TIC_COLORS } from '../../lib/tictac/dataset';
  import ActGrid from '../llm/ActGrid.svelte';

  let {
    corr,
    report,
    threshold,
    onThreshold,
    pca,
    vocab
  }: {
    corr: TensorSnap;
    report: SparsityReport;
    threshold: number;
    onThreshold: (v: number) => void;
    pca: { x: number; y: number }[];
    vocab: string[];
  } = $props();

  const SW = 220;
  const SH = 170;
  const pad = 16;
  const scaled = $derived.by(() => {
    const xs = pca.map((p) => p.x);
    const ys = pca.map((p) => p.y);
    const sx = Math.max(1e-9, ...xs.map(Math.abs));
    const sy = Math.max(1e-9, ...ys.map(Math.abs));
    return pca.map((p) => ({
      x: pad + ((p.x / sx + 1) / 2) * (SW - 2 * pad),
      y: pad + ((p.y / sy + 1) / 2) * (SH - 2 * pad)
    }));
  });
</script>

<div class="circuit scrollbar">
  <label class="f">
    <span class="lbl">|w| threshold</span>
    <input
      type="range"
      min="0"
      max="0.05"
      step="0.002"
      value={threshold}
      oninput={(e) => onThreshold(Number((e.currentTarget as HTMLInputElement).value))}
    />
    <span class="mono readout">{threshold.toFixed(3)}</span>
  </label>
  <div class="counts mono">
    {report.edges} FFN edges survive · <b>{report.liveUnits}</b> units live · {(report.frac * 100).toFixed(0)}% of all weights pruned
  </div>

  <div class="sub mono faint">unit × feature correlation (Pearson r over the probe suite)</div>
  <ActGrid matrix={corr} signed colLabel="features" />
  <div class="faint hint">columns: X line-threats ×8 · O line-threats ×8 · to-move · cell occupancy ×9. A unit hugging one threat column is a candidate line detector.</div>

  <div class="sub mono faint">token embeddings — top-2 PCs</div>
  <svg viewBox="0 0 {SW} {SH}" class="pcasvg">
    {#each scaled as p, i (i)}
      <circle cx={p.x} cy={p.y} r="4" fill={TIC_COLORS[vocab[i]] ?? '#8b93a7'} />
      <text x={p.x + 6} y={p.y + 3} class="mono ptxt">{vocab[i]}</text>
    {/each}
  </svg>
  <div class="faint hint">corners blue · edges orange · center yellow — orbit-mates clustering is the D₄ structure showing up in embedding geometry.</div>
</div>

<style>
  .circuit { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .readout { font-size: 11px; color: var(--muted); }
  .counts { font-size: 11.5px; }
  .counts b { color: var(--model); }
  .sub { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
  .hint { font-size: 10.5px; line-height: 1.4; }
  .pcasvg { width: 100%; max-width: 300px; background: var(--bg-2); border-radius: 6px; flex: 0 0 auto; }
  .ptxt { font-size: 9px; fill: var(--muted); }
</style>
