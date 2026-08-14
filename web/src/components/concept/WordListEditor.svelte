<script lang="ts">
  /**
   * The word-list input panel shared by the morphology and morfessor lenses:
   * sample picker, one-word-per-line textarea, and a lens-specific hint line
   * (passed as a snippet so it can carry inline markup).
   */
  import type { Snippet } from 'svelte';

  let {
    text = $bindable(),
    sampleKey = $bindable(),
    samples,
    hint
  }: {
    text: string;
    sampleKey: string;
    samples: Record<string, string>;
    hint: Snippet;
  } = $props();

  function pick(k: string) {
    sampleKey = k;
    text = samples[k];
  }
</script>

<div class="editor">
  <label class="f">
    <span class="lbl">sample</span>
    <select value={sampleKey} onchange={(e) => pick((e.currentTarget as HTMLSelectElement).value)}>
      {#each Object.keys(samples) as k}<option value={k}>{k}</option>{/each}
    </select>
  </label>
  <textarea class="word-input mono scrollbar" bind:value={text} spellcheck="false"
            placeholder="one word per line, optional “word count”…"></textarea>
  <p class="editor-hint faint">{@render hint()}</p>
</div>

<style>
  /* Word-list editor lives in its own panel; the textarea fills it. */
  .editor { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  .editor .f { flex: 0 0 auto; }
  .word-input {
    flex: 1 1 auto; min-height: 60px; width: 100%; box-sizing: border-box;
    font-size: 12px; padding: 7px 9px; line-height: 1.5; resize: none; white-space: pre;
  }
  .editor-hint { flex: 0 0 auto; font-size: 11px; margin: 0; line-height: 1.4; white-space: normal; }
</style>
