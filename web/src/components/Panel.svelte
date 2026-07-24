<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../lib/panels/panels.svelte';

  let {
    manager,
    id,
    /** Flex grow weight when expanded and sharing space with siblings. */
    weight = 1,
    /** Always size to content (e.g. the short cost readout), never grow. */
    fit = false,
    children,
    /** Optional right-aligned meta in the title bar (counts, step #, legend). */
    actions
  }: {
    manager: PanelManager;
    id: string;
    weight?: number;
    fit?: boolean;
    children: Snippet;
    actions?: Snippet;
  } = $props();

  const state = $derived(manager.get(id));
  const title = $derived(state?.title ?? id);
  const collapsed = $derived(state?.collapsed ?? false);
  const focused = $derived(manager.isFocused(id));

  // Focused panels are positioned by CSS (overlay); otherwise flex sizing.
  const flexStyle = $derived(
    focused
      ? ''
      : collapsed || fit
        ? 'flex:0 0 auto;'
        : `flex:${weight} 1 0;min-height:0;`
  );
</script>

<section class="panel" class:collapsed class:focused style={flexStyle}>
  <header class="pbar">
    <button
      class="ptitle"
      onclick={() => manager.toggleCollapse(id)}
      title={collapsed ? 'Expand' : 'Collapse'}
    >
      <span class="caret" class:open={!collapsed}>▸</span>
      <span class="ptext">{title}</span>
    </button>

    <div class="pright">
      {#if actions}<span class="pmeta">{@render actions()}</span>{/if}
      <button
        class="picon"
        onclick={() => manager.toggleFocus(id)}
        title={focused ? 'Restore (Esc)' : 'Maximize'}
        aria-label={focused ? 'Restore panel' : 'Maximize panel'}
      >
        {focused ? '🗗' : '⤢'}
      </button>
    </div>
  </header>

  {#if !collapsed}
    <div class="pbody">{@render children()}</div>
  {/if}
</section>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden; /* clip body to the rounded corners */
  }

  /* Maximize: overlay the whole PanelHost. */
  .panel.focused {
    position: absolute;
    inset: 0;
    z-index: 50;
    margin: 0;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
  }

  .pbar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 6px 4px 4px;
    background: var(--panel-2);
    border-bottom: 1px solid var(--border);
    user-select: none;
  }

  .ptitle {
    display: flex;
    align-items: center;
    gap: 5px;
    background: transparent;
    border: none;
    border-radius: var(--r-sm);
    padding: 3px 5px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    min-width: 0;
  }
  .ptitle:hover { background: var(--panel); color: var(--text); }
  .ptext { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .caret {
    display: inline-block;
    font-size: 9px;
    color: var(--faint);
    transition: transform 0.12s ease;
    transform: rotate(0deg);
  }
  .caret.open { transform: rotate(90deg); }

  .pright { display: flex; align-items: center; gap: 4px; flex: 0 0 auto; }
  .pmeta { font-size: 11px; color: var(--muted); white-space: nowrap; }

  .picon {
    background: transparent;
    border: none;
    border-radius: var(--r-sm);
    padding: 2px 6px;
    font-size: 12px;
    line-height: 1;
    color: var(--faint);
  }
  .picon:hover { background: var(--panel); color: var(--text); }

  .pbody {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 10px 12px;
  }
</style>
