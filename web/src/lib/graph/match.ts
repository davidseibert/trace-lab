/**
 * SUBDUE's beam, made tractable for toy graphs.
 *
 * Real SUBDUE grows substructures one edge at a time and counts instances by
 * subgraph isomorphism over arbitrarily large graphs. The structure-finding is
 * the hard, exponential part. Here the graphs are small (≤ ~40 nodes) and the
 * lesson is the MDL trade-off, not the mining, so we take a direct, exact route:
 *
 *   1. Enumerate every *connected vertex subset* of the current graph up to
 *      `maxNodes`. The induced subgraph on that subset is one candidate instance.
 *   2. Group subsets by canonical key (see canon.ts) — each group is one
 *      substructure together with all the places it occurs.
 *   3. Within a group, greedily pick a maximal *vertex-disjoint* set of instances
 *      — the graph analogue of the string lens replacing non-overlapping digrams.
 *      Keep the substructure only if ≥2 disjoint instances survive.
 *
 * Induced (not arbitrary) subgraphs are the right teaching choice: an instance is
 * a chunk you literally cut out of the graph and replace with one node, so what
 * you fold must be exactly the edges that live inside that chunk.
 *
 * Everything is bounded (`maxNodes`, `maxEdges`, a hard subset cap) and
 * deterministic (stable enumeration + sort), so the precomputed trace is
 * reproducible and the UI stays a pure function of the step index.
 */

import type { GraphModel, PatEdge } from './graph';
import { canonicalize } from './canon';

export interface Embedding {
  /** Graph node ids, in canonical position order (pattern pos p → graph node). */
  nodes: number[];
}

export interface PatternGroup {
  key: string;
  /** Node labels in canonical order. */
  patNodes: number[];
  /** Internal edges, indexing into patNodes, in canonical order. */
  patEdges: PatEdge[];
  /** A maximal vertex-disjoint instance set (≥2). */
  instances: Embedding[];
  /** Heuristic compression potential, for capping the candidate list. */
  score: number;
}

/** Hard ceiling on how many connected subsets we enumerate, as a safety net. */
const SUBSET_CAP = 20000;

/**
 * Find every repeated, vertex-disjoint substructure in the current graph.
 * Returns the top `cap` groups by compression potential.
 */
export function findPatterns(
  m: GraphModel,
  maxNodes: number,
  maxEdges: number,
  cap: number
): PatternGroup[] {
  const N = m.nodes.length;

  // Undirected adjacency drives connectivity-preserving growth; the directed,
  // labeled edges are what we actually fold.
  const adj: Set<number>[] = Array.from({ length: N }, () => new Set<number>());
  for (const e of m.edges) {
    if (e.src === e.dst) continue; // self-loops never define a multi-node motif
    adj[e.src].add(e.dst);
    adj[e.dst].add(e.src);
  }

  // 1. Enumerate connected vertex subsets (size 2..maxNodes) by BFS growth,
  //    deduped by their sorted membership. Complete: every connected subset of
  //    size s is some size-(s-1) connected subset plus one boundary vertex, so
  //    level-by-level growth from singletons reaches each subset (many ways —
  //    hence the `seen` dedup, which also makes each expand exactly once).
  //    Worst case this is exponential in maxNodes (dense graphs have ~N·d^(s-1)
  //    subsets per level); SUBSET_CAP is the tractability guarantee.
  const seen = new Set<string>();
  const subsets: number[][] = [];
  let frontier: number[][] = [];
  for (let i = 0; i < N; i++) frontier.push([i]);

  while (frontier.length > 0 && subsets.length < SUBSET_CAP) {
    const next: number[][] = [];
    for (const sub of frontier) {
      if (sub.length >= 2) subsets.push(sub);
      if (sub.length >= maxNodes) continue;
      const members = new Set(sub);
      const neighbors = new Set<number>();
      for (const v of sub) for (const u of adj[v]) if (!members.has(u)) neighbors.add(u);
      for (const u of neighbors) {
        const grown = [...sub, u].sort((a, b) => a - b);
        const k = grown.join(',');
        if (seen.has(k)) continue;
        seen.add(k);
        next.push(grown);
        if (seen.size >= SUBSET_CAP) break;
      }
      if (seen.size >= SUBSET_CAP) break;
    }
    frontier = next;
  }

  // 2. Canonicalize each subset's induced subgraph and group by key. Two
  //    induced subgraphs are isomorphic (labels included) iff their canonical
  //    keys are equal, so a Map replaces pairwise isomorphism tests: O(subsets)
  //    canonicalizations instead of O(subsets²) matchings. This is where the
  //    "subgraph matching" actually happens.
  const groups = new Map<
    string,
    { patNodes: number[]; patEdges: PatEdge[]; embeddings: Embedding[] }
  >();

  for (const ids of subsets) {
    const local = new Map<number, number>();
    ids.forEach((id, i) => local.set(id, i));

    const localEdges: PatEdge[] = [];
    for (const e of m.edges) {
      const s = local.get(e.src);
      const d = local.get(e.dst);
      if (s !== undefined && d !== undefined && s !== d) {
        localEdges.push({ src: s, dst: d, label: e.label });
      }
    }
    if (localEdges.length === 0 || localEdges.length > maxEdges) continue;

    const labels = ids.map((id) => m.nodes[id].label);
    const { key, order } = canonicalize(labels, localEdges);

    // `order` maps canonical position → local index; `pos` is its inverse, so
    // edges can be rewritten into pattern-position coordinates.
    const pos = new Array(ids.length);
    order.forEach((orig, p) => (pos[orig] = p));

    const embedding: Embedding = { nodes: order.map((orig) => ids[orig]) };

    let g = groups.get(key);
    if (!g) {
      const patNodes = order.map((orig) => labels[orig]);
      const patEdges = localEdges
        .map((e) => ({ src: pos[e.src], dst: pos[e.dst], label: e.label }))
        .sort((a, b) => a.src - b.src || a.dst - b.dst || a.label - b.label);
      g = { patNodes, patEdges, embeddings: [] };
      groups.set(key, g);
    }
    g.embeddings.push(embedding);
  }

  // 3. Per group: greedily select a maximal vertex-disjoint instance set.
  //    MAXIMAL (nothing more fits), not maximum — the true optimum is max
  //    independent set in the instance-overlap graph, NP-hard, and classic
  //    SUBDUE greedes here too. `apply()` in graph.ts assumes disjointness:
  //    each node belongs to at most one folded instance.
  const out: PatternGroup[] = [];
  for (const [key, g] of groups) {
    if (g.embeddings.length < 2) continue;

    // Deterministic order: by the instance's node-id tuple.
    const sorted = [...g.embeddings].sort((a, b) => {
      const an = [...a.nodes].sort((x, y) => x - y);
      const bn = [...b.nodes].sort((x, y) => x - y);
      for (let i = 0; i < an.length; i++) if (an[i] !== bn[i]) return an[i] - bn[i];
      return 0;
    });

    const used = new Set<number>();
    const instances: Embedding[] = [];
    for (const emb of sorted) {
      if (emb.nodes.some((id) => used.has(id))) continue;
      for (const id of emb.nodes) used.add(id);
      instances.push(emb);
    }
    if (instances.length < 2) continue;

    out.push({
      key,
      patNodes: g.patNodes,
      patEdges: g.patEdges,
      instances,
      // More instances and more folded edges ⇒ more bits removed from the graph.
      score: instances.length * (g.patEdges.length + 1)
    });
  }

  out.sort((a, b) => b.score - a.score);
  return out.slice(0, cap);
}
