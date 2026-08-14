/**
 * Pure layout model for the panel system — types, defaults, persistence schema,
 * and the hydrate() reconciler. No runes, no DOM: everything here is plain data
 * so PanelManager stays a thin reactive wrapper and this file stays testable.
 *
 * Persisted layouts are never trusted: hydrate() drops unknown panel ids,
 * re-seats missing ones at their default location, clamps every number, and
 * falls back to the code-owned defaults on any parse or structure failure.
 */

export interface PanelDef {
  id: string;
  /** Shown in the title bar and the sidebar rail. */
  title: string;
  /** Start collapsed (e.g. reference panels). Reset returns here, not to open. */
  collapsed?: boolean;
  /** Always size to content (e.g. the short cost readout), never grow. */
  fit?: boolean;
  /** Opt into the zoom controls + ctrl-wheel (text/table content). */
  zoomable?: boolean;
}

/** The lens's default arrangement — code-owned, the target of reset(). */
export interface LayoutDef {
  /** Main-area columns, left→right; inner arrays are panel ids top→bottom. */
  columns: string[][];
  /** Column flex weights (parallel to `columns`); default all-1. */
  widths?: number[];
  /** Per-panel size weight along its container's main axis; default 1. */
  weights?: Record<string, number>;
  /** Panels starting in the right rail / bottom dock (normally empty). */
  sidebar?: string[];
  bottom?: string[];
  /** Bottom dock height as a fraction of the host; default 0.3. */
  bottomHeight?: number;
}

export interface PanelState {
  id: string;
  title: string;
  /** Body hidden, only the title bar remains; siblings reclaim the space. */
  collapsed: boolean;
  /** Share of the container's main axis (axis-agnostic flex weight). */
  weight: number;
  /** Content zoom factor (font-size multiplier), 1 = 100%. */
  zoom: number;
}

export interface ColumnState {
  width: number;
  panels: string[];
}

/** The full mutable layout — what PanelManager holds and what we persist. */
export interface LayoutState {
  columns: ColumnState[];
  sidebar: string[];
  bottom: string[];
  bottomHeight: number;
  collapsed: Record<string, boolean>;
  weights: Record<string, number>;
  zoom: Record<string, number>;
  focusedId: string | null;
}

/** v2 localStorage shape. v1 (unversioned) was {collapsed, focusedId}. */
export interface PersistedV2 extends LayoutState {
  v: 2;
}

export const MAX_COLUMNS = 4;
export const MIN_COL_PX = 160;
export const MIN_PANEL_PX = 90;
export const ZOOM_MIN = 0.5;
export const ZOOM_MAX = 2.5;
export const DOCK_MIN = 0.12;
export const DOCK_MAX = 0.6;

export const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

const num = (x: unknown, fallback: number): number =>
  typeof x === 'number' && Number.isFinite(x) && x > 0 ? x : fallback;

export function defaultState(defs: PanelDef[], layout: LayoutDef): LayoutState {
  const collapsed: Record<string, boolean> = {};
  const weights: Record<string, number> = {};
  const zoom: Record<string, number> = {};
  for (const d of defs) {
    collapsed[d.id] = d.collapsed ?? false;
    weights[d.id] = layout.weights?.[d.id] ?? 1;
    zoom[d.id] = 1;
  }
  return {
    columns: layout.columns.map((panels, i) => ({
      width: layout.widths?.[i] ?? 1,
      panels: [...panels]
    })),
    sidebar: [...(layout.sidebar ?? [])],
    bottom: [...(layout.bottom ?? [])],
    bottomHeight: clamp(layout.bottomHeight ?? 0.3, DOCK_MIN, DOCK_MAX),
    collapsed,
    weights,
    zoom,
    focusedId: null
  };
}

export function cloneState(s: LayoutState): LayoutState {
  return {
    columns: s.columns.map((c) => ({ width: c.width, panels: [...c.panels] })),
    sidebar: [...s.sidebar],
    bottom: [...s.bottom],
    bottomHeight: s.bottomHeight,
    collapsed: { ...s.collapsed },
    weights: { ...s.weights },
    zoom: { ...s.zoom },
    focusedId: s.focusedId
  };
}

const EPS = 0.01;
const near = (a: number, b: number) => Math.abs(a - b) < EPS;
const sameIds = (a: string[], b: string[]) =>
  a.length === b.length && a.every((x, i) => x === b[i]);

/** Structural equality (numbers within ε) — drives isDirty. */
export function sameState(a: LayoutState, b: LayoutState): boolean {
  if (a.focusedId !== b.focusedId) return false;
  if (!near(a.bottomHeight, b.bottomHeight)) return false;
  if (a.columns.length !== b.columns.length) return false;
  for (let i = 0; i < a.columns.length; i++) {
    if (!near(a.columns[i].width, b.columns[i].width)) return false;
    if (!sameIds(a.columns[i].panels, b.columns[i].panels)) return false;
  }
  if (!sameIds(a.sidebar, b.sidebar) || !sameIds(a.bottom, b.bottom)) return false;
  for (const id of Object.keys(a.collapsed)) {
    if (a.collapsed[id] !== b.collapsed[id]) return false;
    if (!near(a.weights[id] ?? 1, b.weights[id] ?? 1)) return false;
    if (!near(a.zoom[id] ?? 1, b.zoom[id] ?? 1)) return false;
  }
  return true;
}

/**
 * Build a live LayoutState from defs + defaults + whatever is in storage.
 * Handles: nothing stored, v1 blobs, stale ids after code changes, junk.
 */
export function hydrate(
  defs: PanelDef[],
  layout: LayoutDef,
  raw: string | null
): LayoutState {
  const dflt = defaultState(defs, layout);
  if (!raw) return dflt;

  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(raw);
    if (!obj || typeof obj !== 'object') return dflt;
  } catch {
    return dflt;
  }

  const known = new Set(defs.map((d) => d.id));
  const validFocus = (id: unknown): string | null =>
    typeof id === 'string' && known.has(id) ? id : null;

  // ---- v1 migration: only collapse state and focus carry over -------------
  if ((obj as { v?: unknown }).v !== 2) {
    const st = cloneState(dflt);
    const oldCollapsed = (obj as { collapsed?: Record<string, unknown> }).collapsed;
    if (oldCollapsed && typeof oldCollapsed === 'object') {
      for (const d of defs) {
        if (typeof oldCollapsed[d.id] === 'boolean')
          st.collapsed[d.id] = oldCollapsed[d.id] as boolean;
      }
    }
    st.focusedId = validFocus((obj as { focusedId?: unknown }).focusedId);
    return st;
  }

  // ---- v2 reconciliation --------------------------------------------------
  try {
    const p = obj as Partial<PersistedV2>;
    const placed = new Set<string>();
    const take = (ids: unknown): string[] =>
      Array.isArray(ids)
        ? ids.filter((id): id is string => {
            if (typeof id !== 'string' || !known.has(id) || placed.has(id)) return false;
            placed.add(id);
            return true;
          })
        : [];

    let columns: ColumnState[] = Array.isArray(p.columns)
      ? p.columns.map((c) => ({
          width: num((c as ColumnState)?.width, 1),
          panels: take((c as ColumnState)?.panels)
        }))
      : [];

    const sidebar = take(p.sidebar);
    const bottom = take(p.bottom);

    // Any def id the stored layout doesn't know goes to its default location.
    for (const d of defs) {
      if (placed.has(d.id)) continue;
      if (dflt.sidebar.includes(d.id)) sidebar.push(d.id);
      else if (dflt.bottom.includes(d.id)) bottom.push(d.id);
      else {
        const dc = dflt.columns.findIndex((c) => c.panels.includes(d.id));
        const at = Math.min(Math.max(dc, 0), Math.max(columns.length - 1, 0));
        if (!columns.length) columns.push({ width: 1, panels: [] });
        columns[at].panels.push(d.id);
      }
      placed.add(d.id);
    }

    columns = columns.filter((c) => c.panels.length > 0).slice(0, MAX_COLUMNS);
    // A layout with no visible main area is junk — start over.
    if (!columns.length && !bottom.length) return dflt;

    const collapsed: Record<string, boolean> = {};
    const weights: Record<string, number> = {};
    const zoom: Record<string, number> = {};
    for (const d of defs) {
      collapsed[d.id] =
        typeof p.collapsed?.[d.id] === 'boolean' ? p.collapsed[d.id] : dflt.collapsed[d.id];
      weights[d.id] = num(p.weights?.[d.id], dflt.weights[d.id]);
      zoom[d.id] = clamp(num(p.zoom?.[d.id], 1), ZOOM_MIN, ZOOM_MAX);
    }

    return {
      columns,
      sidebar,
      bottom,
      bottomHeight: clamp(num(p.bottomHeight, dflt.bottomHeight), DOCK_MIN, DOCK_MAX),
      collapsed,
      weights,
      zoom,
      focusedId: validFocus(p.focusedId)
    };
  } catch {
    return dflt;
  }
}
