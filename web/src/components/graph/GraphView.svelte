<script lang="ts">
  import type { ScoredMove } from '../../lib/mdl/types';
  import {
    type GraphModel,
    type SubMove,
    labelToken,
    isBaseLabel
  } from '../../lib/graph/graph';
  import { labelColor } from '../../lib/graph/colors';
  import type { Pt } from '../../lib/graph/layout';

  let {
    model,
    layout,
    chosen
  }: {
    model: GraphModel;
    /** Position per BASE node id, in [0,1]². */
    layout: Pt[];
    chosen: ScoredMove<SubMove> | null;
  } = $props();

  const PAD = 12;
  const sx = (x: number) => PAD + x * (100 - 2 * PAD);
  const sy = (y: number) => PAD + y * (100 - 2 * PAD);

  // Each current node sits at the centroid of the base nodes it expands to, so a
  // collapsed motif lands where its parts were and nothing jumps as you scrub.
  const pos = $derived(
    model.nodes.map((nd) => {
      let x = 0;
      let y = 0;
      for (const id of nd.originIds) {
        x += layout[id]?.x ?? 0.5;
        y += layout[id]?.y ?? 0.5;
      }
      const n = nd.originIds.length || 1;
      return { x: sx(x / n), y: sy(y / n) };
    })
  );

  const radius = (i: number) => (isBaseLabel(model, model.nodes[i].label) ? 3 : 4.4);

  // Which nodes/edges are about to be folded by the chosen move.
  const instOf = $derived.by(() => {
    const map = new Map<number, number>();
    if (chosen) chosen.move.instances.forEach((inst, k) => inst.forEach((id) => map.set(id, k)));
    return map;
  });
  const hotNode = (i: number) => instOf.has(i);
  const hotEdge = (s: number, d: number) =>
    instOf.has(s) && instOf.get(s) === instOf.get(d);

  // Trim an edge to the destination node's rim so the arrowhead sits outside it.
  function endpoint(s: number, d: number) {
    const a = pos[s];
    const b = pos[d];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy) || 1;
    const r = radius(d) + 1.4;
    return { x2: b.x - (dx / dist) * r, y2: b.y - (dy / dist) * r };
  }
  const mid = (s: number, d: number) => ({
    x: (pos[s].x + pos[d].x) / 2,
    y: (pos[s].y + pos[d].y) / 2
  });

  const showEdgeLabels = $derived(model.edgeLabels.length > 1 && model.edges.length <= 36);
</script>

<div class="graph-view">
  <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" role="img" aria-label="graph">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
              orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="var(--faint)" />
      </marker>
      <marker id="arrow-hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
              orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="var(--chosen)" />
      </marker>
    </defs>

    {#each model.edges as e}
      {@const ep = endpoint(e.src, e.dst)}
      {@const hot = hotEdge(e.src, e.dst)}
      <line
        x1={pos[e.src].x} y1={pos[e.src].y} x2={ep.x2} y2={ep.y2}
        class="edge" class:hot
        marker-end={hot ? 'url(#arrow-hot)' : 'url(#arrow)'}
      />
      {#if showEdgeLabels}
        {@const m2 = mid(e.src, e.dst)}
        <text x={m2.x} y={m2.y} class="elabel" dy="-0.6">{model.edgeLabels[e.label]}</text>
      {/if}
    {/each}

    {#each model.nodes as nd, i}
      {@const c = labelColor(model, nd.label)}
      {@const isSub = !isBaseLabel(model, nd.label)}
      <g class="node" class:hot={hotNode(i)}>
        <circle cx={pos[i].x} cy={pos[i].y} r={radius(i)}
                fill={c.bg} stroke={hotNode(i) ? 'var(--chosen)' : c.border}
                stroke-width={isSub ? 1.1 : 0.7} class:subring={isSub} />
        <text x={pos[i].x} y={pos[i].y} dy="0.32em" class="nlabel" fill={c.fg}>{labelToken(model, nd.label)}</text>
      </g>
    {/each}
  </svg>

  {#if chosen}
    <p class="next-note muted">
      Next: fold the {chosen.move.instances.length}× highlighted
      <span class="mono" style="color:var(--chosen)">{chosen.label}</span> into one node.
    </p>
  {:else}
    <p class="next-note good">Converged — no substructure repays its dictionary entry.</p>
  {/if}
</div>

<style>
  .graph-view { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; }
  svg {
    flex: 1 1 auto; min-height: 0; width: 100%; height: auto;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r-sm);
  }
  .edge { stroke: var(--border-2); stroke-width: 0.5; vector-effect: non-scaling-stroke; }
  .edge.hot { stroke: var(--chosen); stroke-width: 1; }
  .elabel {
    font-family: var(--mono); font-size: 2.2px; fill: var(--faint);
    text-anchor: middle; user-select: none;
  }
  .nlabel {
    font-family: var(--mono); font-size: 2.8px; font-weight: 700;
    text-anchor: middle; user-select: none; pointer-events: none;
  }
  .subring { filter: drop-shadow(0 0 1px var(--border-2)); }
  .node.hot .nlabel { fill: var(--text); }
  .next-note { flex: 0 0 auto; font-size: 12.5px; margin: 0; }
</style>
