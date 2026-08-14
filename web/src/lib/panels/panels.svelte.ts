/**
 * Panel management — the shared chrome every lens sits inside. v2: on top of
 * collapse/restore and the single "focus" (maximize) slot, the manager now owns
 * the whole arrangement — main-area columns with widths, per-panel weights,
 * a sidebar rail (with a "peek" overlay slot), a bottom dock, and per-panel
 * content zoom — all persisted per lens and reconciled against the code-owned
 * defaults in lib/panels/layout.ts.
 *
 * Like `Player`, this is a runes class so the UI is a pure function of its
 * state: every component reads `manager` state and re-renders when it changes.
 */

import {
  type PanelDef,
  type LayoutDef,
  type PanelState,
  type ColumnState,
  type LayoutState,
  type PersistedV2,
  defaultState,
  cloneState,
  sameState,
  hydrate,
  clamp,
  MAX_COLUMNS,
  ZOOM_MIN,
  ZOOM_MAX,
  DOCK_MIN,
  DOCK_MAX
} from './layout';

export type { PanelDef, LayoutDef, PanelState, ColumnState };

/** Where a dragged panel can land. */
export type DropTarget =
  | { kind: 'column'; col: number; index: number }
  | { kind: 'newColumn'; at: number }
  | { kind: 'sidebar' }
  | { kind: 'bottom'; index: number };

export class PanelManager {
  /** id -> state. Deeply reactive, so mutating `.collapsed` is enough. */
  states = $state<Record<string, PanelState>>({});
  /** Main-area columns, left→right. */
  columns = $state<ColumnState[]>([]);
  /** Panels tucked into the right rail. */
  sidebar = $state<string[]>([]);
  /** Panels in the full-width bottom dock. */
  bottom = $state<string[]>([]);
  /** Dock height as a fraction of the host. */
  bottomHeight = $state(0.3);
  /** The single maximized panel, or null. */
  focusedId = $state<string | null>(null);
  /** The sidebar panel currently peeked as an overlay, or null. Not persisted. */
  peekId = $state<string | null>(null);

  /** Stable registration order (nav-independent identity for defensive reads). */
  order: string[] = [];

  #storageKey: string;
  #defs: Map<string, PanelDef>;
  #layoutDef: LayoutDef;
  #defaults: LayoutState;

  constructor(key: string, defs: PanelDef[], layout: LayoutDef) {
    this.#storageKey = `trace-lab:panels:${key}`;
    this.#defs = new Map(defs.map((d) => [d.id, d]));
    this.#layoutDef = layout;
    this.#defaults = defaultState(defs, layout);
    this.order = defs.map((d) => d.id);

    let raw: string | null = null;
    if (typeof localStorage !== 'undefined') {
      try {
        raw = localStorage.getItem(this.#storageKey);
      } catch {
        raw = null;
      }
    }
    this.#apply(hydrate(defs, layout, raw));
  }

  // ---- reads ---------------------------------------------------------------

  get(id: string): PanelState | undefined {
    return this.states[id];
  }

  def(id: string): PanelDef | undefined {
    return this.#defs.get(id);
  }

  isFocused(id: string): boolean {
    return this.focusedId === id;
  }

  /** True when any overlay (maximize or peek) is up — drag is disabled then. */
  get hasOverlay(): boolean {
    return this.focusedId !== null || this.peekId !== null;
  }

  /** True when the layout differs from the default (so a Reset button can show). */
  get isDirty(): boolean {
    return !sameState(this.#snapshot(), this.#defaults);
  }

  // ---- collapse / focus (v1 API, unchanged) --------------------------------

  toggleCollapse(id: string): void {
    const s = this.states[id];
    if (!s) return;
    s.collapsed = !s.collapsed;
    // A collapsed panel can't also be the maximized one.
    if (s.collapsed && this.focusedId === id) this.focusedId = null;
    this.#save();
  }

  focus(id: string): void {
    const s = this.states[id];
    if (!s) return;
    s.collapsed = false; // maximizing implies expanded
    this.focusedId = id;
    this.#save();
  }

  unfocus(): void {
    this.focusedId = null;
    this.#save();
  }

  toggleFocus(id: string): void {
    this.focusedId === id ? this.unfocus() : this.focus(id);
  }

  /** Back to the code-owned default layout, nothing maximized, zoom 100%. */
  reset(): void {
    this.peekId = null;
    this.#apply(cloneState(this.#defaults));
    this.#save();
  }

  // ---- moving panels -------------------------------------------------------

  movePanel(id: string, target: DropTarget): void {
    if (!this.states[id]) return;
    const from = this.#remove(id);

    switch (target.kind) {
      case 'column': {
        let { col, index } = target;
        col = clamp(col, 0, Math.max(this.columns.length - 1, 0));
        if (!this.columns.length) {
          this.columns.push({ width: 1, panels: [] });
          col = 0;
        }
        // Removing from the same column above the drop point shifts it up one.
        if (from?.kind === 'column' && from.col === col && from.index < index) index--;
        const panels = this.columns[col].panels;
        panels.splice(clamp(index, 0, panels.length), 0, id);
        break;
      }
      case 'newColumn': {
        if (this.columns.length >= MAX_COLUMNS) {
          // Shouldn't be offered as a target; land in the nearest column instead.
          const col = clamp(target.at, 0, this.columns.length - 1);
          this.columns[col].panels.push(id);
          break;
        }
        const avg =
          this.columns.reduce((s, c) => s + c.width, 0) / (this.columns.length || 1);
        this.columns.splice(clamp(target.at, 0, this.columns.length), 0, {
          width: avg || 1,
          panels: [id]
        });
        break;
      }
      case 'sidebar': {
        this.sidebar.push(id);
        if (this.focusedId === id) this.focusedId = null;
        break;
      }
      case 'bottom': {
        let { index } = target;
        if (from?.kind === 'bottom' && from.index < index) index--;
        this.bottom.splice(clamp(index, 0, this.bottom.length), 0, id);
        break;
      }
    }

    this.#prune();
    this.#save();
  }

  /** Bring a rail panel back to its default column (clamped), appended. */
  restoreFromSidebar(id: string): void {
    if (!this.sidebar.includes(id)) return;
    if (this.peekId === id) this.peekId = null;
    const dc = this.#defaults.columns.findIndex((c) => c.panels.includes(id));
    const col = clamp(dc, 0, Math.max(this.columns.length - 1, 0));
    this.movePanel(id, { kind: 'column', col, index: this.columns[col]?.panels.length ?? 0 });
  }

  togglePeek(id: string): void {
    this.peekId = this.peekId === id ? null : id;
  }

  // ---- resizing ------------------------------------------------------------

  setColumnWidths(widths: number[]): void {
    this.columns.forEach((c, i) => {
      if (Number.isFinite(widths[i]) && widths[i] > 0) c.width = widths[i];
    });
    this.#save();
  }

  setPanelWeights(ids: string[], weights: number[]): void {
    ids.forEach((id, i) => {
      const s = this.states[id];
      if (s && Number.isFinite(weights[i]) && weights[i] > 0) s.weight = weights[i];
    });
    this.#save();
  }

  setBottomHeight(frac: number): void {
    this.bottomHeight = clamp(frac, DOCK_MIN, DOCK_MAX);
    this.#save();
  }

  /** dblclick on a column gutter: restore the pair's default widths. */
  resetGutterColumns(i: number): void {
    const a = this.columns[i];
    const b = this.columns[i + 1];
    if (!a || !b) return;
    const sameShape = this.columns.length === this.#defaults.columns.length;
    const da = sameShape ? this.#defaults.columns[i]?.width : undefined;
    const db = sameShape ? this.#defaults.columns[i + 1]?.width : undefined;
    if (da && db) {
      a.width = da;
      b.width = db;
    } else {
      const mid = (a.width + b.width) / 2;
      a.width = mid;
      b.width = mid;
    }
    this.#save();
  }

  /** dblclick on an intra-column gutter: restore the pair's default weights. */
  resetGutterPanels(aId: string, bId: string): void {
    const a = this.states[aId];
    const b = this.states[bId];
    if (!a || !b) return;
    a.weight = this.#defaults.weights[aId] ?? 1;
    b.weight = this.#defaults.weights[bId] ?? 1;
    this.#save();
  }

  // ---- zoom ----------------------------------------------------------------

  zoomOf(id: string): number {
    return this.states[id]?.zoom ?? 1;
  }

  setZoom(id: string, z: number): void {
    const s = this.states[id];
    if (!s) return;
    s.zoom = clamp(z, ZOOM_MIN, ZOOM_MAX);
    this.#save();
  }

  zoomBy(id: string, dir: 1 | -1): void {
    this.setZoom(id, this.zoomOf(id) * (dir === 1 ? 1.1 : 1 / 1.1));
  }

  resetZoom(id: string): void {
    this.setZoom(id, 1);
  }

  // ---- internals -----------------------------------------------------------

  /** Remove a panel from whichever container holds it; report where it was. */
  #remove(id: string): { kind: 'column' | 'sidebar' | 'bottom'; col?: number; index: number } | null {
    for (let ci = 0; ci < this.columns.length; ci++) {
      const idx = this.columns[ci].panels.indexOf(id);
      if (idx >= 0) {
        this.columns[ci].panels.splice(idx, 1);
        return { kind: 'column', col: ci, index: idx };
      }
    }
    const si = this.sidebar.indexOf(id);
    if (si >= 0) {
      this.sidebar.splice(si, 1);
      return { kind: 'sidebar', index: si };
    }
    const bi = this.bottom.indexOf(id);
    if (bi >= 0) {
      this.bottom.splice(bi, 1);
      return { kind: 'bottom', index: bi };
    }
    return null;
  }

  #prune(): void {
    for (let i = this.columns.length - 1; i >= 0; i--) {
      if (!this.columns[i].panels.length) this.columns.splice(i, 1);
    }
  }

  #apply(st: LayoutState): void {
    this.columns = st.columns;
    this.sidebar = st.sidebar;
    this.bottom = st.bottom;
    this.bottomHeight = st.bottomHeight;
    this.focusedId = st.focusedId;
    const states: Record<string, PanelState> = {};
    for (const id of this.order) {
      const d = this.#defs.get(id)!;
      states[id] = {
        id,
        title: d.title,
        collapsed: st.collapsed[id] ?? d.collapsed ?? false,
        weight: st.weights[id] ?? 1,
        zoom: st.zoom[id] ?? 1
      };
    }
    this.states = states;
  }

  #snapshot(): LayoutState {
    const collapsed: Record<string, boolean> = {};
    const weights: Record<string, number> = {};
    const zoom: Record<string, number> = {};
    for (const id of this.order) {
      const s = this.states[id];
      collapsed[id] = s?.collapsed ?? false;
      weights[id] = s?.weight ?? 1;
      zoom[id] = s?.zoom ?? 1;
    }
    return {
      columns: this.columns.map((c) => ({ width: c.width, panels: [...c.panels] })),
      sidebar: [...this.sidebar],
      bottom: [...this.bottom],
      bottomHeight: this.bottomHeight,
      collapsed,
      weights,
      zoom,
      focusedId: this.focusedId
    };
  }

  #save(): void {
    if (typeof localStorage === 'undefined') return;
    const data: PersistedV2 = { v: 2, ...this.#snapshot() };
    try {
      localStorage.setItem(this.#storageKey, JSON.stringify(data));
    } catch {
      /* storage full / blocked — layout just won't persist, no harm. */
    }
  }
}
