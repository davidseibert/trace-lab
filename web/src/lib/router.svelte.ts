/**
 * Hash router — `#/<lensId>?k=v`. Hash-based on purpose: the built app is
 * static files behind any server (vite dev, docker nginx), so there's no
 * fallback rewrite to configure, and swapping to history-API paths later is a
 * contained change.
 *
 * Two kinds of URL state, two write paths:
 *  - the PATH changes via plain `<a href="#/lens">` links (a real history
 *    entry, so back/forward walks between lenses);
 *  - the QUERY is each lens's live settings, written with `replaceState` so
 *    scrubbing a knob doesn't spam history. Lenses read params once at mount
 *    (the component remounts on every path change) and write them back from an
 *    `$effect`, which makes every refresh — and every copied URL — land on the
 *    same lens with the same settings.
 */

export class Router {
  /** '' = the index page; otherwise a lens id. */
  path = $state('');
  params = $state(new URLSearchParams());
  /**
   * Bumped on every EXTERNAL navigation (link click, back/forward, hand-edited
   * URL) — setQuery uses replaceState, which fires no hashchange, so it never
   * bumps. The shell keys the routed component on path+epoch, so navigating to
   * the same lens with different params remounts it and the params are read.
   */
  epoch = $state(0);

  constructor() {
    if (typeof window === 'undefined') return;
    this.#read();
    window.addEventListener('hashchange', () => this.#read());
  }

  #read() {
    const h = window.location.hash.replace(/^#\/?/, '');
    const qi = h.indexOf('?');
    this.path = decodeURIComponent(qi >= 0 ? h.slice(0, qi) : h);
    this.params = new URLSearchParams(qi >= 0 ? h.slice(qi + 1) : '');
    this.epoch++;
  }

  get(k: string): string | null {
    return this.params.get(k);
  }

  num(k: string): number | null {
    const v = this.params.get(k);
    if (v === null || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  bool(k: string): boolean | null {
    const v = this.params.get(k);
    return v === null ? null : v !== '0' && v !== 'false';
  }

  /**
   * Replace the current query string wholesale. Pass `null`/`undefined` for a
   * default-valued setting to keep it out of the URL, so URLs stay short and a
   * bare `#/lens` link always means "factory settings".
   */
  setQuery(rec: Record<string, string | number | boolean | null | undefined>) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(rec)) {
      if (v === null || v === undefined || v === '') continue;
      q.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
    }
    const qs = q.toString();
    history.replaceState(null, '', `#/${this.path}${qs ? `?${qs}` : ''}`);
    this.params = q;
  }
}

export const router = new Router();
