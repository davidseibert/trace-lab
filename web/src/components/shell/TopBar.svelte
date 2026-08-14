<script lang="ts">
  /**
   * The one topbar every lens uses: consistent chrome (padding, gap, wrap, the
   * panel-layout reset button), lens-specific controls as children. The small
   * control classes (.f, .lbl, .cb, .formula, …) are global in app.css so the
   * controls look identical across lenses without each one re-styling them.
   */
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../../lib/panels/panels.svelte';

  let { panels, children }: { panels?: PanelManager; children: Snippet } = $props();
</script>

<div class="topbar panel">
  {@render children()}

  {#if panels?.isDirty}
    <button class="ghost reset-layout" onclick={() => panels!.reset()} title="Reset panel layout">⤢ reset</button>
  {/if}
</div>

<style>
  .topbar {
    display: flex;
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    padding: 7px 12px;
    flex-wrap: wrap;
  }
  .reset-layout { padding: 4px 9px; font-size: 11px; white-space: nowrap; }
</style>
