<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../lib/panels/panels.svelte';
  import type { DragController } from '../lib/panels/drag.svelte';

  let {
    manager,
    id,
    children,
    /** Optional right-aligned meta in the title bar (counts, step #, legend). */
    actions,
    /**
     * Main axis of the container this panel sits in: 'v' for a column,
     * 'h' for the bottom dock, 'stack' for the small-screen stacked mode.
     */
    axis = 'v',
    /** Drag controller from PanelHost; absent = this panel isn't draggable. */
    drag,
    /** Rendered as the sidebar peek overlay: restore/close instead of maximize. */
    peek = false
  }: {
    manager: PanelManager;
    id: string;
    children: Snippet;
    actions?: Snippet;
    axis?: 'v' | 'h' | 'stack';
    drag?: DragController;
    peek?: boolean;
  } = $props();

  const st = $derived(manager.get(id));
  const def = $derived(manager.def(id));
  const title = $derived(st?.title ?? id);
  const collapsed = $derived(st?.collapsed ?? false);
  const focused = $derived(manager.isFocused(id));
  const weight = $derived(st?.weight ?? 1);
  const zoom = $derived(st?.zoom ?? 1);

  // Focused panels are positioned by CSS (overlay); peeked ones fill their
  // overlay wrapper; otherwise flex sizing along the container's axis.
  // `fit` panels size to content and never grow.
  const flexStyle = $derived(
    focused
      ? ''
      : peek
        ? 'flex:1 1 auto;min-height:0;'
        : collapsed || def?.fit
          ? 'flex:0 0 auto;'
          : axis === 'stack'
            ? 'flex:0 0 auto;max-height:70vh;'
            : axis === 'h'
              ? `flex:${weight} 1 0;min-width:0;`
              : `flex:${weight} 1 0;min-height:0;`
  );

  function onTitleClick() {
    if (drag?.consumeClick()) return; // a drag just ended; not a click
    manager.toggleCollapse(id);
  }

  // ---- content zoom (opt-in via def.zoomable) ------------------------------
  let bodyEl = $state<HTMLElement | null>(null);

  // Svelte 5 registers `onwheel` attributes passively — preventDefault (needed
  // to keep ctrl+wheel from browser-zooming the page) requires a manual
  // non-passive listener.
  $effect(() => {
    const el = bodyEl;
    if (!el || !def?.zoomable) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      manager.zoomBy(id, e.deltaY < 0 ? 1 : -1);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  });

  /**
   * Zoom so the content's width matches the panel's. Measures the body's
   * first child (the scroll container) against its content; since font-size
   * zoom scales content linearly, one step converges both directions.
   */
  function fitWidth() {
    const scroller = bodyEl?.firstElementChild as HTMLElement | null;
    if (!scroller) return;
    const inner = (scroller.firstElementChild as HTMLElement | null) ?? scroller;
    const avail = scroller.clientWidth;
    const content = Math.max(inner.scrollWidth, inner.getBoundingClientRect().width);
    if (avail > 0 && content > 0) {
      manager.setZoom(id, manager.zoomOf(id) * (avail / content) * 0.98);
    }
  }
</script>

<section
  class="panel"
  class:collapsed
  class:focused
  class:drag-source={drag?.dragging === id}
  style={flexStyle}
  data-panel={id}
>
  <header
    class="pbar"
    class:draggable={!!drag && !peek}
    role="toolbar"
    tabindex={-1}
    onpointerdown={(e) => (peek ? undefined : drag?.down(e, id))}
  >
    {#if peek}
      <span class="ptitle static"><span class="ptext">{title}</span></span>
    {:else}
      <button
        class="ptitle"
        onclick={onTitleClick}
        title={collapsed ? 'Expand' : 'Collapse'}
      >
        <span class="caret" class:open={!collapsed}>▸</span>
        <span class="ptext">{title}</span>
      </button>
    {/if}

    <div class="pright">
      {#if actions}<span class="pmeta">{@render actions()}</span>{/if}

      {#if def?.zoomable && !collapsed}
        <span class="zoomctl">
          <button class="picon" onclick={() => manager.zoomBy(id, -1)} title="Zoom out (Ctrl+scroll)">−</button>
          <button
            class="pzoom mono"
            onclick={() => manager.resetZoom(id)}
            title="Reset zoom to 100%"
          >{Math.round(zoom * 100)}%</button>
          <button class="picon" onclick={() => manager.zoomBy(id, 1)} title="Zoom in (Ctrl+scroll)">+</button>
          <button class="picon" onclick={fitWidth} title="Fit content width">⤡</button>
        </span>
      {/if}

      {#if peek}
        <button
          class="picon"
          onclick={() => manager.restoreFromSidebar(id)}
          title="Restore to layout"
          aria-label="Restore panel to layout"
        >⇤</button>
        <button
          class="picon"
          onclick={() => manager.togglePeek(id)}
          title="Close (Esc)"
          aria-label="Close peek"
        >✕</button>
      {:else}
        <button
          class="picon"
          onclick={() => manager.toggleFocus(id)}
          title={focused ? 'Restore (Esc)' : 'Maximize'}
          aria-label={focused ? 'Restore panel' : 'Maximize panel'}
        >
          {focused ? '🗗' : '⤢'}
        </button>
      {/if}
    </div>
  </header>

  {#if !collapsed}
    <div class="pbody" style="--zoom:{zoom}" bind:this={bodyEl}>{@render children()}</div>
  {/if}
</section>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden; /* clip body to the rounded corners */
  }

  /* The source panel stays in place, dimmed, while its ghost is dragged. */
  .panel.drag-source { opacity: 0.4; }

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
  .pbar.draggable { cursor: grab; touch-action: none; }

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
  .ptitle.static { cursor: default; }
  .ptitle.static:hover { background: transparent; color: var(--muted); }
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

  .zoomctl { display: inline-flex; align-items: center; gap: 1px; }

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

  .pzoom {
    background: transparent;
    border: none;
    border-radius: var(--r-sm);
    padding: 2px 4px;
    font-size: 10px;
    line-height: 1;
    color: var(--faint);
    min-width: 34px;
  }
  .pzoom:hover { background: var(--panel); color: var(--text); }

  .pbody {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 10px 12px;
  }
</style>
