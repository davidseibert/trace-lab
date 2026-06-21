<script lang="ts">
  import { type GraphModel, labelToken, vocabSize } from '../../lib/graph/graph';
  import { labelColor } from '../../lib/graph/colors';

  let { model }: { model: GraphModel } = $props();

  // Lay a substructure's nodes on a small circle — deterministic, compact, and
  // enough to read a 3–5 node motif at a glance.
  function ring(n: number): { x: number; y: number }[] {
    if (n === 1) return [{ x: 16, y: 16 }];
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      return { x: 16 + Math.cos(a) * 11, y: 16 + Math.sin(a) * 11 };
    });
  }
</script>

<div class="subs-view">
  <div class="head">
    <span class="panel-title"><span class="swatch" style="background:var(--model)"></span> Dictionary &nbsp;·&nbsp; L(M)</span>
    <span class="muted mono">{model.subs.length} substructure{model.subs.length === 1 ? '' : 's'}</span>
  </div>

  {#if model.subs.length === 0}
    <p class="empty muted">No substructures yet — the model is empty. The whole graph is raw nodes and edges.</p>
  {:else}
    <div class="list scrollbar">
      {#each model.subs as sub, k}
        {@const subId = model.nodeLabels.length + k}
        {@const c = labelColor(model, subId)}
        {@const pts = ring(sub.nodes.length)}
        <div class="sub">
          <div class="sub-head">
            <span class="chip" style="background:{c.bg}; color:{c.fg}; border-color:{c.border}">S{k}</span>
            <svg viewBox="0 0 32 32" class="glyph">
              {#each sub.edges as e}
                <line x1={pts[e.src].x} y1={pts[e.src].y} x2={pts[e.dst].x} y2={pts[e.dst].y} class="ge" />
              {/each}
              {#each sub.nodes as lab, i}
                {@const nc = labelColor(model, lab)}
                <circle cx={pts[i].x} cy={pts[i].y} r="3.4" fill={nc.bg} stroke={nc.border} stroke-width="0.6" />
                <text x={pts[i].x} y={pts[i].y} dy="0.32em" class="gt" fill={nc.fg}>{labelToken(model, lab)}</text>
              {/each}
            </svg>
          </div>
          <span class="expand mono faint">
            {sub.nodes.map((l) => labelToken(model, l)).join(',')} · {sub.nodes.length}n·{sub.edges.length}e
          </span>
        </div>
      {/each}
    </div>
    <p class="vocab faint mono">node vocab {vocabSize(model)} = {model.nodeLabels.length} base + {model.subs.length} sub</p>
  {/if}
</div>

<style>
  .subs-view { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  .head { display: flex; justify-content: space-between; align-items: center; flex: 0 0 auto; }
  .empty { font-size: 12.5px; margin: 4px 0; }
  .list { display: flex; flex-direction: column; gap: 7px; overflow-y: auto; min-height: 0; flex: 1 1 auto; padding-right: 4px; }
  .sub {
    display: flex; flex-direction: column; gap: 5px; align-items: center;
    padding: 7px 6px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm);
  }
  .sub-head { display: flex; align-items: center; gap: 8px; }
  .glyph { width: 46px; height: 46px; flex: 0 0 auto; background: var(--panel); border: 1px solid var(--border); border-radius: 4px; }
  .ge { stroke: var(--border-2); stroke-width: 0.7; }
  .gt { font-family: var(--mono); font-size: 3.4px; font-weight: 700; text-anchor: middle; user-select: none; }
  .chip {
    font-family: var(--mono); font-size: 12px; font-weight: 700; line-height: 1;
    padding: 4px 6px; border: 1px solid; border-radius: 4px; white-space: pre; user-select: none;
  }
  .expand { font-size: 11px; text-align: center; line-height: 1.3; word-break: break-word; }
  .vocab { flex: 0 0 auto; font-size: 11px; margin: 0; }
</style>
