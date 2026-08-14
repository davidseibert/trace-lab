/**
 * GRAPH LENS — SUBDUE-style substructure compression.
 *
 * The exact graph analogue of the grammar lens. Where grammar names a recurring
 * adjacent *digram* and folds its occurrences into a rule symbol, SUBDUE
 * discovers a recurring connected *subgraph* (a triangle, a ring, a motif),
 * names it, and collapses every non-overlapping instance into a single composite
 * node. The collapsed pattern lives once in the substructure dictionary — that is
 * L(M) — and the graph that remains is L(D|M). Adding a substructure grows the
 * dictionary but shrinks the graph; the trough between them is the MDL story,
 * lifted from a sequence to a structurally richer object.
 *
 *   grammar             →  graph
 *   ----------------------------------------------
 *   terminals (chars)   →  nodeLabels + edgeLabels
 *   Rule { rhs:[a,b] }  →  Substructure (a small internal graph)
 *   sequence: number[]  →  nodes[] / edges[]
 *   digram (a,b)        →  connected subgraph + its instance set
 *   countDigrams        →  findPatterns (canon + enumeration, match.ts)
 *   replaceDigram       →  collapse disjoint instances → composite node
 *
 * A composite node carries a substructure label id, so a later substructure can
 * contain an earlier one — the dictionary composes hierarchically, exactly like
 * grammar building "the" from (t,h)+e.
 *
 * Notably this lens needs NO custom engine: it implements `MdlProblem` and rides
 * the generic `trace()` runner unchanged, unlike the Morfessor and LLM lenses.
 * The SUBDUE beam happens inside `candidates()`; the engine still just picks the
 * argmin and applies it. That is the README's thesis made concrete in a domain
 * far from strings.
 *
 * Two code models, toggled by `codeMode` (same as the grammar lens):
 *   'uniform' — every label reference costs log2(V); every edge names its two
 *               endpoints at log2(N) each plus an edge-label choice.
 *   'shannon' — label references cost -log2(p) by empirical frequency; endpoint
 *               *naming* stays uniform (positions aren't frequency-coded).
 */

import type { CodeMode, CostBreakdown, MdlProblem, ScoredMove } from '../mdl/types';
import { fmt, surprisal, uniformBits } from '../mdl/format';
import { findPatterns } from './match';

export type { CodeMode };

export interface GraphConfig {
  codeMode: CodeMode;
  /** Bits to spell one base label when transmitting the alphabet. Fixed term. */
  charBits: number;
  /** Include the model-of-the-model cost (dictionary framing / code table). */
  includeOverhead: boolean;
  /** Beam caps that keep subgraph mining tractable on toy graphs. */
  maxPatNodes: number;
  maxPatEdges: number;
}

/** An edge inside a substructure pattern; src/dst index into the pattern nodes. */
export interface PatEdge {
  src: number;
  dst: number;
  label: number;
}

/** A learned substructure: a small directed labeled graph. Its node label id is
 *  `nodeLabels.length + k` for the k-th substructure. */
export interface Substructure {
  /** Label ids of the pattern's internal nodes (may reference earlier subs). */
  nodes: number[];
  edges: PatEdge[];
}

export interface GraphNode {
  /** Label id: < nodeLabels.length ⇒ base label; else a substructure symbol. */
  label: number;
  /** Base node ids this node ultimately expands to — used only for a stable,
   *  jump-free layout (a composite sits at the centroid of its constituents). */
  originIds: number[];
}

export interface GraphEdge {
  /** Index into `nodes`. */
  src: number;
  dst: number;
  /** Index into `edgeLabels`. */
  label: number;
}

export interface GraphModel {
  /** Base node-label vocabulary; label id === index. */
  nodeLabels: string[];
  /** Edge-label vocabulary; label id === index. */
  edgeLabels: string[];
  /** Learned dictionary. Substructure k has node-label id nodeLabels.length + k. */
  subs: Substructure[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  config: GraphConfig;
}

/** A move = "name this substructure and fold its disjoint instances". */
export interface SubMove {
  pattern: { nodes: number[]; edges: PatEdge[] };
  /** Each instance = graph node ids in pattern-position order; vertex-disjoint. */
  instances: number[][];
  key: string;
}

export const defaultConfig: GraphConfig = {
  codeMode: 'uniform',
  charBits: 8,
  includeOverhead: true,
  maxPatNodes: 5,
  maxPatEdges: 8
};

// ---------------------------------------------------------------------------
// Label helpers (mirror grammar's symbol helpers)
// ---------------------------------------------------------------------------

/** Node-label vocabulary size: base labels + learned substructures. */
export const vocabSize = (m: GraphModel): number => m.nodeLabels.length + m.subs.length;

export const isBaseLabel = (m: GraphModel, id: number): boolean => id < m.nodeLabels.length;

export const subIndexOf = (m: GraphModel, id: number): number => id - m.nodeLabels.length;

/** Printable token for a node label id: the base label, or "S{k}". */
export function labelToken(m: GraphModel, id: number): string {
  return isBaseLabel(m, id) ? m.nodeLabels[id] : `S${subIndexOf(m, id)}`;
}

/** Compact textual render of a pattern, for the candidates table. */
export function renderPattern(m: GraphModel, p: { nodes: number[]; edges: PatEdge[] }): string {
  const labs = p.nodes.map((id) => labelToken(m, id)).join(',');
  return `{${labs}} ${p.nodes.length}n·${p.edges.length}e`;
}

// ---------------------------------------------------------------------------
// Frequency model (for Shannon coding) — label uses across graph + dictionary
// ---------------------------------------------------------------------------

function labelCounts(m: GraphModel): { node: Map<number, number>; edge: Map<number, number> } {
  const node = new Map<number, number>();
  const edge = new Map<number, number>();
  const bumpN = (id: number) => node.set(id, (node.get(id) ?? 0) + 1);
  const bumpE = (id: number) => edge.set(id, (edge.get(id) ?? 0) + 1);
  for (const nd of m.nodes) bumpN(nd.label);
  for (const ed of m.edges) bumpE(ed.label);
  for (const s of m.subs) {
    for (const id of s.nodes) bumpN(id);
    for (const e of s.edges) bumpE(e.label);
  }
  return { node, edge };
}

// ---------------------------------------------------------------------------
// Cost: L(M) + L(D | M)
// ---------------------------------------------------------------------------

/**
 * Description length of the current state. Uniform-mode counting scheme, for a
 * graph of N nodes, E edges, node vocab Vn = |base labels| + |subs|, edge vocab Ve:
 *
 *   L(D|M)  nodes: N · log2(Vn)                    one label reference per node
 *           edges: E · (2·log2(N) + log2(Ve))      name src, name dst, name label
 *
 * Adjacency is thus an explicit edge list — each endpoint a uniform choice among
 * the N *current* nodes, so collapsing nodes cheapens every remaining edge (and
 * shrinks N·log2(Vn) even as log2(Vn) itself grows with the vocab).
 *
 *   L(M)    each substructure is the same encoding applied to its own small
 *           graph, except a pattern edge's endpoints index only the pattern's
 *           own n nodes: log2(n) bits, not log2(N). Plus one label ref per sub
 *           (framing) and the fixed alphabet spell-out.
 *
 * Shannon mode swaps every *label* reference for its surprisal -log2(count/total)
 * under the empirical label frequencies (counted across graph + dictionary, so
 * one shared code covers both); endpoint *naming* stays uniform — positions are
 * identities, not repeatable symbols, so frequency-coding them is meaningless.
 */
export function cost(m: GraphModel): CostBreakdown {
  const cfg = m.config;
  const Vn = vocabSize(m);
  const Ve = m.edgeLabels.length;
  const N = m.nodes.length;
  const E = m.edges.length;
  const endpointBits = uniformBits(N); // naming one endpoint among N nodes
  const edgeLabelBits = uniformBits(Ve);
  const alphabetBits = (m.nodeLabels.length + m.edgeLabels.length) * cfg.charBits;

  if (cfg.codeMode === 'uniform') {
    const nodeRefBits = uniformBits(Vn);

    // L(D|M): the compressed graph.
    const graphNodeBits = N * nodeRefBits;
    const graphEdgeBits = E * (2 * endpointBits + edgeLabelBits);
    const dataBits = graphNodeBits + graphEdgeBits;

    // L(M): the substructure dictionary, each encoded as its own small graph.
    let subNodeBits = 0;
    let subEdgeBits = 0;
    for (const s of m.subs) {
      subNodeBits += s.nodes.length * nodeRefBits;
      subEdgeBits += s.edges.length * (2 * uniformBits(s.nodes.length) + edgeLabelBits);
    }
    // Framing: announcing each sub's symbol id costs one label ref — the price
    // of making the dictionary addressable at all.
    const framingBits = cfg.includeOverhead ? m.subs.length * nodeRefBits : 0;
    const modelBits = subNodeBits + subEdgeBits + framingBits + alphabetBits;

    return {
      modelBits,
      dataBits,
      total: modelBits + dataBits,
      modelTerms: [
        {
          label: 'substructure nodes',
          bits: subNodeBits,
          detail: `${m.subs.length} subs, ${fmt(nodeRefBits)} bits/label`
        },
        {
          label: 'substructure edges',
          bits: subEdgeBits,
          detail: `endpoints + label, encoded once per sub`
        },
        ...(cfg.includeOverhead
          ? [
              {
                label: 'dictionary framing',
                bits: framingBits,
                detail: `${m.subs.length} × ${fmt(nodeRefBits)} bits`
              }
            ]
          : []),
        {
          label: 'alphabet',
          bits: alphabetBits,
          detail: `${m.nodeLabels.length}+${m.edgeLabels.length} labels × ${cfg.charBits} bits`,
          fixed: true
        }
      ],
      dataTerms: [
        {
          label: 'graph nodes',
          bits: graphNodeBits,
          detail: `${N} nodes × ${fmt(nodeRefBits)} bits/label`
        },
        {
          label: 'graph edges',
          bits: graphEdgeBits,
          detail: `${E} edges × (2×${fmt(endpointBits)} + ${fmt(edgeLabelBits)}) bits`
        }
      ],
      meta: {
        'bits/label': nodeRefBits,
        'node vocab': Vn,
        nodes: N,
        edges: E,
        subs: m.subs.length
      }
    };
  }

  // Shannon / entropy coding of labels (endpoints stay uniform log2(N)).
  const { node: nodeCnt, edge: edgeCnt } = labelCounts(m);
  let nodeTotal = 0;
  for (const c of nodeCnt.values()) nodeTotal += c;
  let edgeTotal = 0;
  for (const c of edgeCnt.values()) edgeTotal += c;
  const nodeLen = (id: number) => surprisal((nodeCnt.get(id) ?? 0) / (nodeTotal || 1));
  const edgeLen = (id: number) => surprisal((edgeCnt.get(id) ?? 0) / (edgeTotal || 1));

  const graphNodeBits = m.nodes.reduce((s, nd) => s + nodeLen(nd.label), 0);
  const graphEdgeBits = m.edges.reduce((s, ed) => s + 2 * endpointBits + edgeLen(ed.label), 0);
  const dataBits = graphNodeBits + graphEdgeBits;

  let subNodeBits = 0;
  let subEdgeBits = 0;
  for (const s of m.subs) {
    subNodeBits += s.nodes.reduce((acc, id) => acc + nodeLen(id), 0);
    subEdgeBits += s.edges.reduce(
      (acc, e) => acc + 2 * uniformBits(s.nodes.length) + edgeLen(e.label),
      0
    );
  }
  // Transmit the code tables: one frequency per distinct label used. A count
  // lies in [0, total], so each costs log2(total+1) bits — the honesty term
  // that stops Shannon coding from getting its distribution for free.
  const codeTableBits = cfg.includeOverhead
    ? nodeCnt.size * uniformBits(nodeTotal + 1) + edgeCnt.size * uniformBits(edgeTotal + 1)
    : 0;
  const framingBits = cfg.includeOverhead ? m.subs.length * uniformBits(Vn) : 0;
  const modelBits = subNodeBits + subEdgeBits + codeTableBits + framingBits + alphabetBits;
  const bps = N > 0 ? graphNodeBits / N : 0;

  return {
    modelBits,
    dataBits,
    total: modelBits + dataBits,
    modelTerms: [
      { label: 'substructure nodes', bits: subNodeBits, detail: `${m.subs.length} subs, entropy-coded labels` },
      { label: 'substructure edges', bits: subEdgeBits, detail: `endpoints + entropy-coded label` },
      ...(cfg.includeOverhead
        ? [
            { label: 'code table', bits: codeTableBits, detail: `${nodeCnt.size}+${edgeCnt.size} labels` },
            { label: 'dictionary framing', bits: framingBits, detail: `${m.subs.length} × ${fmt(uniformBits(Vn))} bits` }
          ]
        : []),
      {
        label: 'alphabet',
        bits: alphabetBits,
        detail: `${m.nodeLabels.length}+${m.edgeLabels.length} labels × ${cfg.charBits} bits`,
        fixed: true
      }
    ],
    dataTerms: [
      { label: 'graph nodes', bits: graphNodeBits, detail: `${N} nodes, entropy-coded (${fmt(bps)} avg bits)` },
      { label: 'graph edges', bits: graphEdgeBits, detail: `${E} edges, endpoints uniform + entropy-coded label` }
    ],
    meta: {
      'avg bits/label': bps,
      'node vocab': Vn,
      nodes: N,
      edges: E,
      subs: m.subs.length
    }
  };
}

// ---------------------------------------------------------------------------
// Candidate moves + scoring
// ---------------------------------------------------------------------------

export function candidates(m: GraphModel): SubMove[] {
  const groups = findPatterns(m, m.config.maxPatNodes, m.config.maxPatEdges, 40);
  return groups.map((g) => ({
    pattern: { nodes: g.patNodes, edges: g.patEdges },
    instances: g.instances.map((e) => e.nodes),
    key: g.key
  }));
}

export function apply(m: GraphModel, move: SubMove): GraphModel {
  const newSubId = vocabSize(m); // label id for the new composite node
  const subs = [...m.subs, { nodes: move.pattern.nodes, edges: move.pattern.edges }];

  // Map each old node id either to its instance's composite, or to a kept node.
  const instanceOf = new Map<number, number>(); // old node id → instance index
  move.instances.forEach((inst, k) => inst.forEach((id) => instanceOf.set(id, k)));

  const remap = new Map<number, number>(); // old node id → new node id
  const nodes: GraphNode[] = [];

  // Kept (uncollapsed) nodes first, preserving order.
  m.nodes.forEach((nd, id) => {
    if (!instanceOf.has(id)) {
      remap.set(id, nodes.length);
      nodes.push(nd);
    }
  });

  // One composite node per instance, placed at the centroid of its constituents.
  for (const inst of move.instances) {
    const originIds = inst.flatMap((id) => m.nodes[id].originIds);
    const newId = nodes.length;
    nodes.push({ label: newSubId, originIds });
    for (const id of inst) remap.set(id, newId);
  }

  // Rewire edges. Edges internal to a single instance are now stored inside the
  // substructure definition, so we drop them from the graph; everything else is
  // remapped (external edges follow the collapse onto the composite node).
  const edges: GraphEdge[] = [];
  for (const e of m.edges) {
    const ks = instanceOf.get(e.src);
    const kd = instanceOf.get(e.dst);
    if (ks !== undefined && ks === kd) continue; // internal to one instance → fold away
    edges.push({ src: remap.get(e.src)!, dst: remap.get(e.dst)!, label: e.label });
  }

  return { ...m, subs, nodes, edges };
}

/**
 * ΔL for a candidate: apply the move to a copy and re-cost from scratch, so
 * delta = L(after) − L(before) is EXACT — unlike classic SUBDUE's local
 * estimate, this captures every side effect (Vn grows by 1, so log2(Vn) rises
 * for all node refs; N shrinks, so every surviving edge's endpoints cheapen;
 * Shannon frequencies reshuffle). The gain itself: each of the c folded
 * instances trades |P| node refs + its internal edges for ONE composite ref,
 * while the pattern's own encoding is paid once, in L(M) — so ΔL < 0 exactly
 * when c·(savings per instance) outweighs dictionary + vocab growth.
 */
export function scoreMove(m: GraphModel, move: SubMove, baseline: CostBreakdown): ScoredMove<SubMove> {
  const after = cost(apply(m, move));
  const nextIdx = m.subs.length;
  return {
    move,
    label: `S${nextIdx} ← ${renderPattern(m, move.pattern)}`,
    delta: after.total - baseline.total,
    totalAfter: after.total,
    modelBitsAfter: after.modelBits,
    dataBitsAfter: after.dataBits,
    extra: {
      '×': move.instances.length,
      expands: renderPattern(m, move.pattern)
    }
  };
}

// ---------------------------------------------------------------------------
// Input parsing — an edge list, one edge per line
// ---------------------------------------------------------------------------

/**
 * Parse an edge list into the bare graph. One edge per line:
 *
 *     src:NodeLabel  dst:NodeLabel  EDGELABEL
 *
 * A node's label is bound the first time `id:Label` appears; afterwards the bare
 * id may be reused. The edge label is optional (defaults to "rel"). Lines that
 * are blank or start with `#` are ignored. This mirrors morphology's
 * `parseWordList`: forgiving, paste-friendly toy input.
 */
export function parseEdgeList(text: string, config: GraphConfig): GraphModel {
  const nodeLabels: string[] = [];
  const nodeLabelIndex = new Map<string, number>();
  const edgeLabels: string[] = [];
  const edgeLabelIndex = new Map<string, number>();
  const nodeIndex = new Map<string, number>(); // node key → node index
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const labelId = (label: string): number => {
    let id = nodeLabelIndex.get(label);
    if (id === undefined) {
      id = nodeLabels.length;
      nodeLabels.push(label);
      nodeLabelIndex.set(label, id);
    }
    return id;
  };
  const edgeLabelId = (label: string): number => {
    let id = edgeLabelIndex.get(label);
    if (id === undefined) {
      id = edgeLabels.length;
      edgeLabels.push(label);
      edgeLabelIndex.set(label, id);
    }
    return id;
  };
  const nodeRef = (token: string): number => {
    const [key, label] = token.split(':');
    let idx = nodeIndex.get(key);
    if (idx === undefined) {
      idx = nodes.length;
      nodeIndex.set(key, idx);
      nodes.push({ label: labelId(label ?? key), originIds: [idx] });
    } else if (label !== undefined) {
      nodes[idx].label = labelId(label); // a later mention may name an earlier node
    }
    return idx;
  };

  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (line === '' || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 2) continue;
    const src = nodeRef(parts[0]);
    const dst = nodeRef(parts[1]);
    const label = edgeLabelId(parts[2] ?? 'rel');
    edges.push({ src, dst, label });
  }

  return { nodeLabels, edgeLabels, subs: [], nodes, edges, config };
}

// ---------------------------------------------------------------------------
// Problem factory
// ---------------------------------------------------------------------------

/**
 * Cheap correctness anchor (no test framework in this project). On a graph of
 * four identical directed triangles, the first MDL move must be the triangle,
 * folding all 4 instances, and it must strictly reduce the total. Call in dev.
 */
export function selfCheck(): { ok: boolean; msg: string } {
  const tri = (i: number) =>
    `t${i}a:o t${i}b:o e\nt${i}b:o t${i}c:o e\nt${i}c:o t${i}a:o e`;
  const text = [tri(0), tri(1), tri(2), tri(3)].join('\n');
  const m = parseEdgeList(text, defaultConfig);
  const base = cost(m);
  const cands = candidates(m).map((mv) => scoreMove(m, mv, base)).sort((a, b) => a.delta - b.delta);
  const best = cands[0];
  const ok =
    !!best &&
    best.delta < 0 &&
    best.move.instances.length === 4 &&
    best.move.pattern.nodes.length === 3;
  return {
    ok,
    msg: ok
      ? `graph selfCheck ✓ — triangle ×${best.move.instances.length}, Δ ${best.delta.toFixed(1)} bits`
      : `graph selfCheck ✗ — best=${best ? best.label + ' Δ' + best.delta.toFixed(1) + ' ×' + best.move.instances.length : 'none'}`
  };
}

export function graphProblem(text: string, config: GraphConfig): MdlProblem<GraphModel, SubMove> {
  const initial = parseEdgeList(text, config);
  return {
    name: 'Graph substructure compression',
    blurb:
      'Discover a recurring subgraph, name it, and collapse every non-overlapping instance into one node — the SUBDUE move. Dictionary (model) trades off against the compressed graph (data).',
    initialModel: () => initial,
    cost,
    candidates,
    apply,
    scoreMove
  };
}
