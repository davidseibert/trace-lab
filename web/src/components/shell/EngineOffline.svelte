<script lang="ts">
  /**
   * The shared "engine offline" empty state for instrument lenses. One place
   * for the start-it-up instructions instead of each lens re-writing them.
   */
  import { engine } from '../../lib/logit/engine.svelte';

  let {
    what,
    onUp
  }: {
    /** One line: what this lens needs the engine for. */
    what: string;
    /** Called after a successful retry (e.g. auto-run the default prompt). */
    onUp?: () => void;
  } = $props();

  let retrying = $state(false);
  async function retry() {
    retrying = true;
    const ok = await engine.check();
    retrying = false;
    if (ok) onUp?.();
  }
</script>

<div class="panel empty">
  <p class="mono">engine offline</p>
  <p class="faint">{what} Start the engine service:</p>
  <pre class="mono">make up            # web + engine via Docker

# or bare metal, from engine/:
uv run uvicorn main:app --port 5181</pre>
  <p class="faint">
    (set <span class="mono">HF_HOME</span> first to reuse an existing model cache — see README), then
    <button class="ghost" onclick={retry} disabled={retrying}>{retrying ? 'checking…' : 'retry'}</button>
  </p>
</div>

<style>
  .empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .empty pre {
    background: var(--bg-2);
    border: 1px solid var(--border-2);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
  }
</style>
