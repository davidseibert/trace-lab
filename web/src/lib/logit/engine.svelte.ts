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

/** Shown before the first successful /health so pickers aren't empty. Mirrors
 * ALLOWED_MODELS in engine/main.py — /health overwrites it the moment the
 * engine answers, so this only has to be right enough to render a picker
 * offline. Keep the two in step when adding a model. */
const FALLBACK: ModelInfo[] = [
  { name: 'gpt2', kind: 'hub' },
  { name: 'gpt2-medium', kind: 'hub' },
  { name: 'gpt2-large', kind: 'hub' },
  { name: 'Qwen/Qwen2.5-0.5B', kind: 'hub' },
  { name: 'Qwen/Qwen3-0.6B-Base', kind: 'hub' },
  { name: 'Qwen/Qwen3-0.6B', kind: 'hub' },
  { name: 'Qwen/Qwen3-1.7B', kind: 'hub' },
  { name: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', kind: 'hub' },
  { name: 'google/gemma-3-270m-it', kind: 'hub' },
  { name: 'google/gemma-3-1b-it', kind: 'hub' },
  { name: 'meta-llama/Llama-3.2-1B', kind: 'hub' },
  { name: 'meta-llama/Llama-3.2-1B-Instruct', kind: 'hub' },
  { name: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', kind: 'hub' }
];

export class EngineStore {
  /** Device string when reachable, null when the engine is offline/unknown. */
  device = $state<string | null>(null);
  models = $state<ModelInfo[]>(FALLBACK);
  defaultModel = $state('gpt2');
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
      this.defaultModel = h.default;
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
