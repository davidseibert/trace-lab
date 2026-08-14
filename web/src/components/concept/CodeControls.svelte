<script lang="ts">
  /**
   * The code-model controls every concept lens shows in its topbar: the MDL
   * objective, lens-specific controls (children) after it, the uniform/Shannon
   * code toggle, and — for lenses that price the model-of-the-model — the
   * overhead checkbox. All classes used here are global (app.css), so this
   * component is the single source of the shared markup, not of styling.
   */
  import type { Snippet } from 'svelte';

  type CodeMode = 'uniform' | 'shannon';

  let {
    codeMode = $bindable(),
    includeOverhead = $bindable(true),
    formulaTitle,
    overhead = true,
    overheadTitle,
    children
  }: {
    codeMode: CodeMode;
    includeOverhead?: boolean;
    /** Tooltip spelling out what L(M) and L(D|M) mean for this lens. */
    formulaTitle: string;
    /** Render the overhead checkbox (off for lenses without the toggle). */
    overhead?: boolean;
    overheadTitle?: string;
    /** Lens-specific controls rendered between the formula and the toggle. */
    children?: Snippet;
  } = $props();
</script>

<span class="formula mono" title={formulaTitle}>
  <b style="color:var(--total)">min</b>
  <b style="color:var(--model)">L(M)</b>+<b style="color:var(--data)">L(D|M)</b>
</span>

{@render children?.()}

<div class="f">
  <span class="lbl">code</span>
  <div class="toggle-group">
    <button class:active={codeMode === 'uniform'} onclick={() => (codeMode = 'uniform')}>log₂V</button>
    <button class:active={codeMode === 'shannon'} onclick={() => (codeMode = 'shannon')}>−log₂p</button>
  </div>
</div>

{#if overhead}
  <label class="cb" title={overheadTitle}>
    <input type="checkbox" bind:checked={includeOverhead} /> overhead
  </label>
{/if}
