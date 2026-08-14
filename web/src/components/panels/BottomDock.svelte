<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PanelManager } from '../../lib/panels/panels.svelte';
  import type { DragController } from '../../lib/panels/drag.svelte';
  import { MIN_PANEL_PX } from '../../lib/panels/layout';
  import { resizePair } from '../../lib/panels/pointer';
  import Panel from '../Panel.svelte';

  let {
    manager,
    snippets,
    actions = {},
    drag
  }: {
    manager: PanelManager;
    snippets: Record<string, Snippet>;
    actions?: Record<string, Snippet>;
    drag?: DragController;
  } = $props();

  let dockEl = $state<HTMLElement | null>(null);

  const resizable = (aId: string, bId: string) => {
    const a = manager.get(aId);
    const b = manager.get(bId);
    return (
      !!a && !!b && !a.collapsed && !b.collapsed &&
      !manager.def(aId)?.fit && !manager.def(bId)?.fit
    );
  };

  function gutterDown(e: PointerEvent, aId: string, bId: string) {
    if (!dockEl || !resizable(aId, bId)) return;
    const aEl = dockEl.querySelector(`[data-panel="${aId}"]`);
    const bEl = dockEl.querySelector(`[data-panel="${bId}"]`);
    const a = manager.get(aId);
    const b = manager.get(bId);
    if (!aEl || !bEl || !a || !b) return;
    e.preventDefault();
    resizePair(e, {
      axis: 'x',
      aPx: aEl.getBoundingClientRect().width,
      bPx: bEl.getBoundingClientRect().width,
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
</script>

<div class="dock" data-dock bind:this={dockEl}>
  {#each manager.bottom as id, i (id)}
    {#if i > 0}
      <div
        class="gutter"
        class:live={resizable(manager.bottom[i - 1], id)}
        onpointerdown={(e) => gutterDown(e, manager.bottom[i - 1], id)}
        ondblclick={() => manager.resetGutterPanels(manager.bottom[i - 1], id)}
        role="separator"
        aria-orientation="vertical"
      ></div>
    {/if}
    <Panel {manager} {id} axis="h" {drag} actions={actions[id]}>
      {@render snippets[id]?.()}
    </Panel>
  {/each}
</div>

<style>
  .dock {
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    overflow: clip;
  }

  .gutter {
    flex: 0 0 8px;
    position: relative;
    touch-action: none;
  }
  .gutter.live { cursor: col-resize; }
  .gutter.live::after {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 3px;
    width: 2px;
    border-radius: 1px;
    background: transparent;
    transition: background 0.12s;
  }
  .gutter.live:hover::after { background: var(--border-2); }
</style>
