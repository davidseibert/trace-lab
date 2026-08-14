/**
 * Panel management — the shared chrome every lens (MDL grammar, morphology, and
 * the transformer) sits inside. Deliberately thin for now: collapse/restore, a
 * single "focus" (maximize) slot, and localStorage persistence. The heavier
 * JetBrains-style docking (drag, split, float, tab groups) comes later, once the
 * transformer panels tell us what the layout actually needs.
 *
 * Like `Player`, this is a runes class so the UI is a pure function of its
 * state: every component reads `manager.get(id)` and re-renders when it changes.
 */

export interface PanelDef {
  id: string;
  /** Shown in the title bar and in the (future) docked-tab strip. */
  title: string;
  /** Start collapsed (e.g. reference panels). Reset returns here, not to open. */
  collapsed?: boolean;
}

export interface PanelState {
  id: string;
  title: string;
  /** Body hidden, only the title bar remains; siblings reclaim the space. */
  collapsed: boolean;
}

/** What we round-trip through localStorage. Titles are code-owned, not stored. */
interface Persisted {
  collapsed: Record<string, boolean>;
  focusedId: string | null;
}

export class PanelManager {
  /** id -> state. Deeply reactive, so mutating `.collapsed` is enough. */
  states = $state<Record<string, PanelState>>({});
  /** Stable display order (registration order), used for the tab strip. */
  order = $state<string[]>([]);
  /** The single maximized panel, or null. */
  focusedId = $state<string | null>(null);

  #storageKey: string;
  #defaults: Record<string, boolean> = {};

  constructor(key: string, defs: PanelDef[]) {
    this.#storageKey = `trace-lab:panels:${key}`;
    const saved = this.#load();

    this.order = defs.map((d) => d.id);
    for (const d of defs) {
      this.#defaults[d.id] = d.collapsed ?? false;
      this.states[d.id] = {
        id: d.id,
        title: d.title,
        collapsed: saved?.collapsed?.[d.id] ?? d.collapsed ?? false
      };
    }
    if (saved?.focusedId && this.states[saved.focusedId]) {
      this.focusedId = saved.focusedId;
    }
  }

  get(id: string): PanelState | undefined {
    return this.states[id];
  }

  /** Panels in display order (skips any unknown ids defensively). */
  get panels(): PanelState[] {
    return this.order.map((id) => this.states[id]).filter(Boolean);
  }

  isFocused(id: string): boolean {
    return this.focusedId === id;
  }

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

  /** Back to the default layout (per-panel defaults), nothing maximized. */
  reset(): void {
    for (const id of this.order) {
      const s = this.states[id];
      if (s) s.collapsed = this.#defaults[id] ?? false;
    }
    this.focusedId = null;
    this.#save();
  }

  /** True when the layout differs from the default (so a Reset button can show). */
  get isDirty(): boolean {
    return (
      this.focusedId !== null ||
      this.panels.some((p) => p.collapsed !== (this.#defaults[p.id] ?? false))
    );
  }

  #save(): void {
    if (typeof localStorage === 'undefined') return;
    const data: Persisted = {
      collapsed: Object.fromEntries(
        this.order.map((id) => [id, this.states[id]?.collapsed ?? false])
      ),
      focusedId: this.focusedId
    };
    try {
      localStorage.setItem(this.#storageKey, JSON.stringify(data));
    } catch {
      /* storage full / blocked — layout just won't persist, no harm. */
    }
  }

  #load(): Persisted | null {
    if (typeof localStorage === 'undefined') return null;
    try {
      const raw = localStorage.getItem(this.#storageKey);
      return raw ? (JSON.parse(raw) as Persisted) : null;
    } catch {
      return null;
    }
  }
}
