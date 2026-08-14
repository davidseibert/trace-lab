<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../../lib/panels/panels.svelte';
  import Panel from '../Panel.svelte';

  let {
    manager,
    snippets,
    actions = {},
    /** True while a panel drag is active — the rail shows as a drop zone. */
    dropActive = false
  }: {
    manager: PanelManager;
    snippets: Record<string, Snippet>;
    actions?: Record<string, Snippet>;
    dropActive?: boolean;
  } = $props();
</script>

{#if manager.sidebar.length || dropActive}
  <div class="rail" class:drop={dropActive}>
    {#each manager.sidebar as id (id)}
      <button
        class="item"
        class:active={manager.peekId === id}
        onclick={() => manager.togglePeek(id)}
        title={manager.get(id)?.title}
      >
        {manager.get(id)?.title}
      </button>
    {/each}
    {#if !manager.sidebar.length}
      <span class="empty-hint">⇥</span>
    {/if}
  </div>
{/if}

{#if manager.peekId && manager.sidebar.includes(manager.peekId)}
  <div class="peek">
    <Panel {manager} id={manager.peekId} peek actions={actions[manager.peekId]}>
      {@render snippets[manager.peekId]?.()}
    </Panel>
  </div>
{/if}

<style>
  .rail {
    flex: 0 0 28px;
    min-height: 0;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 6px;
    padding: 4px 0;
    overflow: hidden;
  }
  .rail.drop {
    outline: 1px dashed var(--border-2);
    outline-offset: -1px;
    border-radius: var(--r-sm);
  }

  .item {
    flex: 0 0 auto;
    writing-mode: vertical-rl;
    max-height: 16ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 8px 3px;
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .item:hover { background: var(--panel-2); color: var(--text); }
  .item.active { border-color: var(--model); color: var(--text); }

  .empty-hint {
    margin: auto;
    color: var(--faint);
    font-size: 12px;
  }

  /* Peek overlay: anchored to the PanelHost, beside the rail. */
  .peek {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 34px;
    width: min(40%, 560px);
    z-index: 50;
    display: flex;
    flex-direction: column;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    border-radius: var(--r);
  }
</style>
