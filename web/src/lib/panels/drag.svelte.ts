/**
 * Panel drag-and-drop — a runes controller owned by PanelHost. A drag starts
 * on a panel title bar (5px threshold so plain clicks still collapse), a
 * geometry snapshot of every drop target is taken once at activation (layout
 * doesn't change mid-drag: the source panel stays in place, dimmed), and the
 * pointer is resolved against that snapshot on every move. Drop calls
 * manager.movePanel(); Escape / pointercancel abandon the drag.
 *
 * PanelHost marks the DOM for the snapshot: data-col on columns, data-panel
 * on panel sections, data-dock on the bottom dock. The sidebar target is a
 * synthetic strip along the host's right edge, so it works when the rail is
 * hidden (empty) too.
 */

import type { PanelManager, DropTarget } from './panels.svelte';
import { MAX_COLUMNS } from './layout';
import { trackPointer } from './pointer';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Indicator extends Rect {
  /** 'line' = insertion caret between panels; 'zone' = region highlight. */
  kind: 'line' | 'zone';
}

const toRect = (r: DOMRect): Rect => ({ x: r.x, y: r.y, w: r.width, h: r.height });
const inRect = (r: Rect, x: number, y: number) =>
  x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;

interface Snap {
  host: Rect;
  cols: { rect: Rect; panels: { id: string; rect: Rect }[] }[];
  dock: { rect: Rect; panels: { id: string; rect: Rect }[] } | null;
  rail: Rect; // right-edge strip (synthetic when the rail is not rendered)
}

const RAIL_W = 36;
const EDGE_W = 14; // half-width of a new-column drop strip
const GAP = 8;

export class DragController {
  /** Id of the panel being dragged, null when idle. */
  dragging = $state<string | null>(null);
  /** Pointer position (viewport px) for the floating ghost chip. */
  x = $state(0);
  y = $state(0);
  /** Current resolved drop target, null when over nothing usable. */
  target = $state<DropTarget | null>(null);
  /** Indicator geometry, host-relative px. */
  indicator = $state<Indicator | null>(null);

  /** Set by PanelHost: true in stacked mode or while an overlay is up. */
  disabled = false;

  #manager: PanelManager;
  #host: HTMLElement | null = null;
  #snap: Snap | null = null;
  #didDrag = false;

  constructor(manager: PanelManager) {
    this.#manager = manager;
  }

  setHost(el: HTMLElement | null): void {
    this.#host = el;
  }

  /**
   * Title-bar click guard: after a real drag, the click that follows pointerup
   * must not toggle collapse. Returns true exactly once per completed drag.
   */
  consumeClick(): boolean {
    const d = this.#didDrag;
    this.#didDrag = false;
    return d;
  }

  /** pointerdown on a panel's title bar. */
  down(e: PointerEvent, id: string): void {
    if (this.disabled || e.button !== 0 || !this.#host) return;
    // Buttons on the right side of the bar (maximize, zoom, actions) keep
    // their own behavior; the title/bar body is the drag handle.
    if ((e.target as Element).closest('.pright')) return;

    let active = false;
    trackPointer(e, {
      onMove: (dx, dy, ev) => {
        if (!active && Math.hypot(dx, dy) > 5) {
          active = true;
          this.#didDrag = true;
          this.#snap = this.#takeSnapshot();
          this.dragging = id;
        }
        if (active) {
          this.x = ev.clientX;
          this.y = ev.clientY;
          this.#resolve(ev.clientX, ev.clientY);
        }
      },
      onEnd: (cancelled) => {
        const target = this.target;
        const wasActive = active;
        this.dragging = null;
        this.target = null;
        this.indicator = null;
        this.#snap = null;
        if (wasActive && !cancelled && target) this.#manager.movePanel(id, target);
      }
    });
  }

  #takeSnapshot(): Snap {
    const host = this.#host!;
    const hostRect = toRect(host.getBoundingClientRect());

    const panelsIn = (el: Element) =>
      [...el.querySelectorAll<HTMLElement>('[data-panel]')].map((p) => ({
        id: p.dataset.panel!,
        rect: toRect(p.getBoundingClientRect())
      }));

    const cols = [...host.querySelectorAll<HTMLElement>('[data-col]')].map((c) => ({
      rect: toRect(c.getBoundingClientRect()),
      panels: panelsIn(c)
    }));

    const dockEl = host.querySelector<HTMLElement>('[data-dock]');
    const dock = dockEl
      ? { rect: toRect(dockEl.getBoundingClientRect()), panels: panelsIn(dockEl) }
      : null;

    return {
      host: hostRect,
      cols,
      dock,
      rail: {
        x: hostRect.x + hostRect.w - RAIL_W,
        y: hostRect.y,
        w: RAIL_W,
        h: hostRect.h
      }
    };
  }

  #rel(r: Rect): Rect {
    const h = this.#snap!.host;
    return { x: r.x - h.x, y: r.y - h.y, w: r.w, h: r.h };
  }

  #resolve(cx: number, cy: number): void {
    const snap = this.#snap;
    if (!snap) return;
    const m = this.#manager;

    // 1. Sidebar rail strip along the right edge.
    if (inRect(snap.rail, cx, cy)) {
      this.target = { kind: 'sidebar' };
      this.indicator = { kind: 'zone', ...this.#rel(snap.rail) };
      return;
    }

    // 2. Bottom dock (or, with no dock yet, a strip along the host bottom).
    if (snap.dock) {
      if (cy >= snap.dock.rect.y) {
        const index = snap.dock.panels.filter(
          (p) => p.id !== this.dragging && p.rect.x + p.rect.w / 2 < cx
        ).length;
        this.target = { kind: 'bottom', index };
        this.indicator = this.#insertLine(snap.dock.panels, snap.dock.rect, index, 'x');
        return;
      }
    } else {
      const strip: Rect = {
        x: snap.host.x,
        y: snap.host.y + snap.host.h - 32,
        w: snap.host.w - RAIL_W,
        h: 32
      };
      if (inRect(strip, cx, cy)) {
        this.target = { kind: 'bottom', index: 0 };
        this.indicator = { kind: 'zone', ...this.#rel(strip) };
        return;
      }
    }

    // 3. New-column strips at each column boundary (incl. outer edges).
    if (snap.cols.length > 0 && snap.cols.length < MAX_COLUMNS) {
      for (let b = 0; b <= snap.cols.length; b++) {
        const bx =
          b === 0
            ? snap.cols[0].rect.x
            : b === snap.cols.length
              ? snap.cols[b - 1].rect.x + snap.cols[b - 1].rect.w
              : (snap.cols[b - 1].rect.x + snap.cols[b - 1].rect.w + snap.cols[b].rect.x) / 2;
        if (Math.abs(cx - bx) <= EDGE_W) {
          // A panel alone in a column dropped beside itself is a no-op; allow it,
          // movePanel handles the churn harmlessly.
          this.target = { kind: 'newColumn', at: b };
          this.indicator = {
            kind: 'zone',
            ...this.#rel({
              x: bx - EDGE_W / 2,
              y: snap.cols[0].rect.y,
              w: EDGE_W,
              h: snap.host.h - (snap.dock ? snap.dock.rect.h + GAP : 0)
            })
          };
          return;
        }
      }
    }

    // 4. Inside a column: insertion index from panel midpoints.
    for (let ci = 0; ci < snap.cols.length; ci++) {
      const col = snap.cols[ci];
      if (cx < col.rect.x || cx > col.rect.x + col.rect.w) continue;
      const index = col.panels.filter(
        (p) => p.id !== this.dragging && p.rect.y + p.rect.h / 2 < cy
      ).length;
      this.target = { kind: 'column', col: ci, index };
      this.indicator = this.#insertLine(col.panels, col.rect, index, 'y');
      return;
    }

    this.target = null;
    this.indicator = null;
  }

  /** A 3px insertion caret before `index` among `panels` (dragged one excluded). */
  #insertLine(
    panels: { id: string; rect: Rect }[],
    container: Rect,
    index: number,
    axis: 'x' | 'y'
  ): Indicator {
    const others = panels.filter((p) => p.id !== this.dragging);
    let at: number;
    if (!others.length) {
      at = axis === 'y' ? container.y : container.x;
    } else if (index === 0) {
      at = (axis === 'y' ? others[0].rect.y : others[0].rect.x) - GAP / 2;
    } else {
      const prev = others[Math.min(index, others.length) - 1].rect;
      at = (axis === 'y' ? prev.y + prev.h : prev.x + prev.w) + GAP / 2;
    }
    const line: Rect =
      axis === 'y'
        ? { x: container.x, y: at - 1.5, w: container.w, h: 3 }
        : { x: at - 1.5, y: container.y, w: 3, h: container.h };
    return { kind: 'line', ...this.#rel(line) };
  }
}
