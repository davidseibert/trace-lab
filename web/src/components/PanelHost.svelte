<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../lib/panels/panels.svelte';

  let { manager, children }: { manager: PanelManager; children: Snippet } =
    $props();

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && manager.focusedId) {
      manager.unfocus();
      e.stopPropagation();
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="panel-host" class:has-focus={!!manager.focusedId}>
  {@render children()}
</div>

<style>
  .panel-host {
    position: relative; /* anchor for a maximized panel's overlay */
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    gap: 8px;
  }

  /* Dim everything behind a maximized panel. */
  .panel-host.has-focus::before {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 40;
    background: rgba(8, 10, 14, 0.55);
    border-radius: var(--r);
    pointer-events: none;
  }

  @media (max-width: 880px) {
    .panel-host { flex-direction: column; }
  }
</style>
