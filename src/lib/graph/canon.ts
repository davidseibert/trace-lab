/**
 * Canonical labeling of small directed, labeled graphs.
 *
 * Two induced subgraphs are *instances of the same substructure* exactly when
 * they are isomorphic — same node labels, same directed labeled edges, up to a
 * renumbering of nodes. To group instances (and to dedupe patterns as they grow)
 * we need a fingerprint that is invariant under that renumbering: the canonical
 * key.
 *
 * For the toy graphs this lens targets, patterns have at most a handful of nodes
 * (`maxPatNodes` ≈ 5), so we can afford the honest brute force: try every
 * permutation of the nodes, serialize the relabeled graph, and keep the
 * lexicographically smallest string. That string is the canonical key, and the
 * permutation that achieved it gives a canonical *ordering* of the nodes — which
 * is what lets every instance reconstruct the identical pattern and map pattern
 * positions back onto its own graph nodes.
 *
 * This is the graph analogue of the string lens's digram identity `(a,b)`: there,
 * the pair *is* its own canonical form; here we have to work for it.
 */

import type { PatEdge } from './graph';

/** All permutations of [0..n-1]. n is tiny (≤ maxPatNodes), so this is cheap. */
function permutations(n: number): number[][] {
  if (n <= 1) return [[...Array(n).keys()]];
  const out: number[][] = [];
  const used = new Array(n).fill(false);
  const cur: number[] = [];
  const rec = () => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

export interface Canon {
  /** Isomorphism-invariant fingerprint. Equal keys ⟺ isomorphic graphs. */
  key: string;
  /** order[p] = original node index sitting at canonical position p. */
  order: number[];
}

/**
 * Canonicalize a labeled directed graph given as node labels + internal edges
 * (edges index into `labels`). Returns the canonical key and the node ordering
 * that achieves it.
 */
export function canonicalize(labels: number[], edges: PatEdge[]): Canon {
  const n = labels.length;
  let best: Canon | null = null;

  for (const order of permutations(n)) {
    // pos[orig] = canonical position of original node `orig`.
    const pos = new Array(n);
    for (let p = 0; p < n; p++) pos[order[p]] = p;

    const nodeStr = order.map((o) => labels[o]).join(',');
    const edgeStr = edges
      .map((e) => `${pos[e.src]}>${pos[e.dst]}:${e.label}`)
      .sort()
      .join(';');
    const key = `${nodeStr}#${edgeStr}`;

    if (best === null || key < best.key) best = { key, order };
  }

  // n === 0 can't happen for our patterns (≥2 nodes), but keep it total.
  return best ?? { key: '', order: [] };
}
