<script lang="ts">
  import { untrack, type Snippet } from 'svelte';
  import type { PanelManager } from '../lib/panels/panels.svelte';
  import { DragController } from '../lib/panels/drag.svelte';
  import { MIN_COL_PX, MIN_PANEL_PX, clamp, DOCK_MIN, DOCK_MAX } from '../lib/panels/layout';
  import { trackPointer, resizePair } from '../lib/panels/pointer';
  import Panel from './Panel.svelte';
  import SidebarRail from './panels/SidebarRail.svelte';
  import BottomDock from './panels/BottomDock.svelte';

  let {
    manager,
    snippets,
    actions = {}
  }: {
    manager: PanelManager;
    /** Panel bodies, keyed by panel id. */
    snippets: Record<string, Snippet>;
    /** Optional title-bar meta snippets, keyed by panel id. */
    actions?: Record<string, Snippet>;
  } = $props();

  let hostEl = $state<HTMLElement | null>(null);

  // The manager identity is fixed for this host's lifetime (lenses remount).
  const drag = new DragController(untrack(() => manager));
  $effect(() => drag.setHost(hostEl));

  // Small screens: one scrolling column instead of viewport-fit columns.
  let stacked = $state(false);
  $effect(() => {
    const mq = matchMedia('(max-width: 880px)');
    const update = () => (stacked = mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  });

  $effect(() => {
    drag.disabled = stacked || manager.hasOverlay;
  });

  /** Stacked order: main columns left→right, then dock, then rail panels. */
  const stackedIds = $derived([
    ...manager.columns.flatMap((c) => c.panels),
    ...manager.bottom,
    ...manager.sidebar
  ]);

  function onKeydown(e: KeyboardEvent) {
    if (e.key !== 'Escape') return;
    if (manager.peekId) {
      manager.peekId = null;
      e.stopPropagation();
    } else if (manager.focusedId) {
      manager.unfocus();
      e.stopPropagation();
    }
  }

  // ---- resize gutters ------------------------------------------------------

  const resizable = (aId: string, bId: string) => {
    const a = manager.get(aId);
    const b = manager.get(bId);
    return (
      !!a && !!b && !a.collapsed && !b.collapsed &&
      !manager.def(aId)?.fit && !manager.def(bId)?.fit
    );
  };

  function colGutterDown(e: PointerEvent, i: number) {
    if (!hostEl) return;
    const cols = hostEl.querySelectorAll('[data-col]');
    const aEl = cols[i];
    const bEl = cols[i + 1];
    const ca = manager.columns[i];
    const cb = manager.columns[i + 1];
    if (!aEl || !bEl || !ca || !cb) return;
    e.preventDefault();
    resizePair(e, {
      axis: 'x',
      aPx: aEl.getBoundingClientRect().width,
      bPx: bEl.getBoundingClientRect().width,
      aWeight: ca.width,
      bWeight: cb.width,
      minPx: MIN_COL_PX,
      apply: (wa, wb) => {
        ca.width = wa;
        cb.width = wb;
      },
      commit: () => manager.setColumnWidths(manager.columns.map((c) => c.width))
    });
  }

  function rowGutterDown(e: PointerEvent, ci: number, aId: string, bId: string) {
    if (!hostEl || !resizable(aId, bId)) return;
    const colEl = hostEl.querySelectorAll('[data-col]')[ci];
    if (!colEl) return;
    const aEl = colEl.querySelector(`[data-panel="${aId}"]`);
    const bEl = colEl.querySelector(`[data-panel="${bId}"]`);
    const a = manager.get(aId);
    const b = manager.get(bId);
    if (!aEl || !bEl || !a || !b) return;
    e.preventDefault();
    resizePair(e, {
      axis: 'y',
      aPx: aEl.getBoundingClientRect().height,
      bPx: bEl.getBoundingClientRect().height,
      aWeight: a.weight,
      bWeight: b.weight,
      minPx: MIN_PANEL_PX,
      apply: (wa, wb) => {
        a.weight = wa;
        b.weight = wb;
      },
      commit: () => manager.setPanelWeights([aId, bId], [a.weight, b.weight])
    });
  }

  function dockGutterDown(e: PointerEvent) {
    const mainEl = hostEl?.querySelector('.main');
    if (!mainEl) return;
    const h = mainEl.getBoundingClientRect().height;
    const start = manager.bottomHeight;
    if (h <= 0) return;
    e.preventDefault();
    trackPointer(e, {
      onMove: (_dx, dy) => {
        manager.bottomHeight = clamp(start - dy / h, DOCK_MIN, DOCK_MAX);
      },
      onEnd: (cancelled) => {
        if (cancelled) manager.bottomHeight = start;
        else manager.setBottomHeight(manager.bottomHeight);
      }
    });
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="panel-host" class:has-overlay={manager.hasOverlay} bind:this={hostEl}>
  {#if stacked}
    <div class="stack scrollbar">
      {#each stackedIds as id (id)}
        <Panel {manager} {id} axis="stack" actions={actions[id]}>
          {@render snippets[id]?.()}
        </Panel>
      {/each}
    </div>
  {:else}
    <div class="hbody">
      <div class="main">
        <div class="main-row">
          {#each manager.columns as col, ci (ci)}
            {#if ci > 0}
              <div
                class="gutter v live"
                onpointerdown={(e) => colGutterDown(e, ci - 1)}
                ondblclick={() => manager.resetGutterColumns(ci - 1)}
                role="separator"
                aria-orientation="vertical"
              ></div>
            {/if}
            <div class="col" data-col={ci} style="flex:{col.width} 1 0">
              {#each col.panels as id, pi (id)}
                {#if pi > 0}
                  <div
                    class="gutter h"
                    class:live={resizable(col.panels[pi - 1], id)}
                    onpointerdown={(e) => rowGutterDown(e, ci, col.panels[pi - 1], id)}
                    ondblclick={() => manager.resetGutterPanels(col.panels[pi - 1], id)}
                    role="separator"
                    aria-orientation="horizontal"
                  ></div>
                {/if}
                <Panel {manager} {id} {drag} actions={actions[id]}>
                  {@render snippets[id]?.()}
                </Panel>
              {/each}
            </div>
          {/each}
        </div>

        {#if manager.bottom.length}
          <div
            class="gutter h live dock-gutter"
            onpointerdown={dockGutterDown}
            ondblclick={() => manager.setBottomHeight(0.3)}
            role="separator"
            aria-orientation="horizontal"
          ></div>
          <div class="dock-wrap" style="flex:0 0 {(manager.bottomHeight * 100).toFixed(2)}%">
            <BottomDock {manager} {snippets} {actions} {drag} />
          </div>
        {/if}
      </div>

      <SidebarRail {manager} {snippets} {actions} dropActive={drag.dragging !== null} />
    </div>
  {/if}

  {#if drag.dragging}
    {#if drag.indicator}
      <div
        class="indicator {drag.indicator.kind}"
        style="left:{drag.indicator.x}px;top:{drag.indicator.y}px;width:{drag.indicator.w}px;height:{drag.indicator.h}px"
      ></div>
    {/if}
    <div class="ghost" style="left:{drag.x + 10}px;top:{drag.y + 12}px">
      {manager.get(drag.dragging)?.title}
    </div>
  {/if}
</div>

<style>
  .panel-host {
    position: relative; /* anchor for maximized/peeked panel overlays */
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: clip;
  }

  /* Dim everything behind a maximized or peeked panel. */
  .panel-host.has-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 40;
    background: rgba(8, 10, 14, 0.55);
    border-radius: var(--r);
    pointer-events: none;
  }

  .hbody {
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    gap: 6px;
    overflow: clip;
  }

  .main {
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: clip;
  }

  .main-row {
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    overflow: clip;
  }

  .col {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    overflow: clip;
  }

  .dock-wrap {
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: clip;
  }

  /* Gutters replace the old flex gap: 8px hit area, 2px accent on hover. */
  .gutter {
    flex: 0 0 8px;
    position: relative;
    touch-action: none;
  }
  .gutter.v.live { cursor: col-resize; }
  .gutter.h.live { cursor: row-resize; }
  .gutter.live::after {
    content: '';
    position: absolute;
    border-radius: 1px;
    background: transparent;
    transition: background 0.12s;
  }
  .gutter.v.live::after { top: 0; bottom: 0; left: 3px; width: 2px; }
  .gutter.h.live::after { left: 0; right: 0; top: 3px; height: 2px; }
  .gutter.live:hover::after { background: var(--border-2); }

  /* Drag feedback */
  .indicator {
    position: absolute;
    z-index: 60;
    pointer-events: none;
  }
  .indicator.line { background: var(--model); border-radius: 2px; }
  .indicator.zone {
    background: rgba(91, 156, 255, 0.12);
    border: 1px dashed var(--model);
    border-radius: var(--r-sm);
  }

  .ghost {
    position: fixed;
    z-index: 70;
    pointer-events: none;
    background: var(--panel-2);
    border: 1px solid var(--border-2);
    border-radius: var(--r-sm);
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
  }

  /* Small screens: a single scrolling column; panels take natural heights. */
  .stack {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
  }
</style>
