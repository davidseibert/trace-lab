/**
 * Deterministic force-directed layout for the BASE graph.
 *
 * The trace is precomputed and the UI is a pure function of the step index, so
 * the layout must be deterministic too — a live, randomized force simulation
 * recomputed each frame would make nodes jitter as you scrub, wrecking the
 * debugger feel. So we run a fixed number of Fruchterman–Reingold iterations once
 * over the original graph, seeded from the existing `mulberry32` stream, and
 * freeze the result.
 *
 * Composite nodes (created when a substructure collapses) are never laid out
 * here: the view places each at the centroid of the base nodes it expands to
 * (`GraphNode.originIds`), so a collapsed motif lands exactly where its parts
 * were. Positions therefore stay put across every step.
 *
 * Output coordinates are normalized to [0,1]² — the view scales them into its
 * viewBox with padding.
 */

import { mulberry32, type Rng } from '../llm/rng';

export interface Pt {
  x: number;
  y: number;
}

/** Cheap deterministic string hash → seed, so different graphs differ but a
 *  given graph is always laid out identically. */
export function hashSeed(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Lay out `n` nodes connected by undirected `edges`, returning a position per
 * base node id. `iters` Fruchterman–Reingold steps with linear cooling, then a
 * hard overlap-removal relaxation so motifs (especially disconnected ones) read
 * as visually separate clusters rather than piling on top of each other.
 */
export function forceLayout(
  n: number,
  edges: [number, number][],
  seed: number,
  iters = 400
): Pt[] {
  if (n === 0) return [];
  if (n === 1) return [{ x: 0.5, y: 0.5 }];

  const rng: Rng = mulberry32(seed);
  const pos: Pt[] = Array.from({ length: n }, () => ({ x: rng(), y: rng() }));

  // Fruchterman–Reingold force model: repulsion f_r(d) = REP·k²/d between EVERY
  // pair (O(n²) per iteration — fine at toy sizes), attraction f_a(d) = d²/k
  // along edges. Setting f_r = f_a gives equilibrium spacing d = k·REP^⅓, so
  // k = √(area/n) tiles n nodes evenly over the unit square and REP > 1 pads
  // that spacing so disconnected motifs drift apart. Each node moves along its
  // net force, but at most `temp` per step; the linear cool → 0 is simulated
  // annealing minus the randomness (big early rearrangements, then settling).
  const k = Math.sqrt(1 / n); // ideal edge length
  const REP = 1.7; // repulsion dominates so clusters fly apart
  let temp = 0.18; // max displacement per step, cooled toward 0
  const cool = temp / (iters + 1);
  const eps = 1e-4;

  for (let it = 0; it < iters; it++) {
    const disp: Pt[] = Array.from({ length: n }, () => ({ x: 0, y: 0 }));

    // Repulsion between every pair.
    for (let v = 0; v < n; v++) {
      for (let u = v + 1; u < n; u++) {
        let dx = pos[v].x - pos[u].x;
        let dy = pos[v].y - pos[u].y;
        let dist = Math.hypot(dx, dy);
        if (dist < eps) {
          dx = (rng() - 0.5) * eps;
          dy = (rng() - 0.5) * eps;
          dist = eps;
        }
        const f = (REP * k * k) / dist;
        const ux = (dx / dist) * f;
        const uy = (dy / dist) * f;
        disp[v].x += ux;
        disp[v].y += uy;
        disp[u].x -= ux;
        disp[u].y -= uy;
      }
    }

    // Attraction along edges.
    for (const [a, b] of edges) {
      if (a === b) continue;
      const dx = pos[a].x - pos[b].x;
      const dy = pos[a].y - pos[b].y;
      const dist = Math.hypot(dx, dy) || eps;
      const f = (dist * dist) / k;
      const ux = (dx / dist) * f;
      const uy = (dy / dist) * f;
      disp[a].x -= ux;
      disp[a].y -= uy;
      disp[b].x += ux;
      disp[b].y += uy;
    }

    // Displace, capped by temperature.
    for (let v = 0; v < n; v++) {
      const d = Math.hypot(disp[v].x, disp[v].y) || eps;
      pos[v].x += (disp[v].x / d) * Math.min(d, temp);
      pos[v].y += (disp[v].y / d) * Math.min(d, temp);
    }
    temp -= cool;
  }

  // Normalize UNIFORMLY (same scale on both axes, then center) into [0,1]².
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of pos) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const offX = (span - (maxX - minX)) / 2;
  const offY = (span - (maxY - minY)) / 2;
  for (const p of pos) {
    p.x = (p.x - minX + offX) / span;
    p.y = (p.y - minY + offY) / span;
  }

  // Hard overlap removal AS THE FINAL STEP, in the [0,1] box. FR alone (and the
  // normalize above) can leave node disks coincident; this guarantees a minimum
  // pairwise gap. We clamp into the box each pass rather than renormalizing, so
  // the gap survives in the coordinates the view actually draws. The gap is
  // generous enough that disks plus a little air never touch (≈minSep×76 ≫ the
  // node diameter once mapped to the viewBox).
  const minSep = Math.min(0.13, 0.85 / Math.sqrt(n));
  for (let pass = 0; pass < 300; pass++) {
    let moved = false;
    for (let v = 0; v < n; v++) {
      for (let u = v + 1; u < n; u++) {
        let dx = pos[v].x - pos[u].x;
        let dy = pos[v].y - pos[u].y;
        let dist = Math.hypot(dx, dy);
        if (dist >= minSep) continue;
        if (dist < eps) {
          dx = (rng() - 0.5) * eps;
          dy = (rng() - 0.5) * eps;
          dist = eps;
        }
        const shift = (minSep - dist) / 2;
        const ux = (dx / dist) * shift;
        const uy = (dy / dist) * shift;
        pos[v].x = Math.max(0, Math.min(1, pos[v].x + ux));
        pos[v].y = Math.max(0, Math.min(1, pos[v].y + uy));
        pos[u].x = Math.max(0, Math.min(1, pos[u].x - ux));
        pos[u].y = Math.max(0, Math.min(1, pos[u].y - uy));
        moved = true;
      }
    }
    if (!moved) break;
  }

  return pos;
}
