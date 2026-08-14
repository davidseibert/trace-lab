/**
 * Shared engine-health store for the instrument lenses (Logit·real,
 * Reason·trace, Train·real) and the index page. One place owns "is the engine
 * up, on what device, with which models" instead of each lens re-implementing
 * checkHealth + fallback model lists.
 *
 * A module singleton survives lens switches, so hopping between instruments
 * doesn't re-probe unless asked.
 */
import { fetchHealth, type ModelInfo } from './api';

/** Placeholder so pickers aren't empty before the first /health reply — the
 * engine is the source of truth for the model list, and check() replaces this
 * the moment it answers. */
const FALLBACK: ModelInfo[] = [{ name: 'gpt2', kind: 'hub' }];

export class EngineStore {
  /** Device string when reachable, null when the engine is offline/unknown. */
  device = $state<string | null>(null);
  models = $state<ModelInfo[]>(FALLBACK);
  /** True once any /health has answered (so "offline" isn't shown pre-probe). */
  probed = $state(false);

  up = $derived(this.device !== null);
  hub = $derived(this.models.filter((m) => m.kind === 'hub'));
  local = $derived(this.models.filter((m) => m.kind === 'local'));

  info(name: string): ModelInfo | undefined {
    return this.models.find((m) => m.name === name);
  }

  async check(): Promise<boolean> {
    try {
      const h = await fetchHealth();
      this.device = h.device;
      this.models = h.models;
      return true;
    } catch {
      this.device = null;
      return false;
    } finally {
      this.probed = true;
    }
  }

  /** A request other than /health succeeded — the engine is evidently up. */
  markUp() {
    if (this.device === null) this.device = 'up';
  }
}

export const engine = new EngineStore();
