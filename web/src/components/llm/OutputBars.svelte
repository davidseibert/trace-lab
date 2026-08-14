<script lang="ts">
  let {
    probs,
    vocab,
    colors,
    predId,
    targetToken
  }: {
    probs: Float64Array;
    vocab: string[];
    colors: string[];
    predId: number;
    targetToken: string;
  } = $props();

  // Rank tokens by probability, highest first.
  const ranked = $derived.by(() =>
    vocab
      .map((tok, id) => ({ tok, id, p: probs[id] }))
      .sort((a, b) => b.p - a.p)
  );
  const max = $derived(Math.max(1e-6, ...probs));
</script>

<div class="out scrollbar">
  {#each ranked as { tok, id, p } (id)}
    <div class="orow" class:pick={id === predId}>
      <span class="otok mono" style="color:{colors[id]}">{tok}</span>
      <div class="obar-track">
        <div class="obar" style="width:{(p / max) * 100}%; background:{colors[id]}"></div>
      </div>
      <span class="opct mono">{(p * 100).toFixed(1)}%</span>
      <span class="otag">
        {#if id === predId}<span class="tag pick">PICK</span>{/if}
        {#if tok === targetToken}<span class="tag tgt" title="the correct continuation">✓</span>{/if}
      </span>
    </div>
  {/each}
</div>

<style>
  .out { display: flex; flex-direction: column; gap: 3px; overflow: auto; min-height: 0; flex: 1 1 auto; }
  .orow {
    display: grid;
    grid-template-columns: 64px 1fr 48px 38px;
    gap: 8px;
    align-items: center;
    padding: 2px 4px;
    border-radius: 4px;
  }
  .orow.pick { background: rgba(255, 209, 102, 0.1); outline: 1px solid rgba(255, 209, 102, 0.3); }
  .otok { font-size: 12px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .obar-track { height: 12px; background: var(--bg-2); border-radius: 3px; overflow: hidden; }
  .obar { height: 100%; border-radius: 3px; transition: width 0.2s ease; min-width: 1px; }
  .opct { font-size: 11px; text-align: right; color: var(--muted); }
  .otag { display: flex; gap: 3px; justify-content: flex-end; }
  .tag { font-size: 9px; font-weight: 700; padding: 1px 4px; border-radius: 3px; letter-spacing: 0.04em; }
  .tag.pick { background: var(--chosen); color: #1a1400; }
  .tag.tgt { background: var(--good); color: #04140d; }
</style>
