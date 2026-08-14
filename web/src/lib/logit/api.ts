/**
 * Client for the engine service (engine/ — FastAPI over torch): the real-model
 * counterpart of the mini-GPT lens. The engine runs the classic logit lens and
 * the J-lens over a HuggingFace model and reports everything in the house
 * currency: per-layer code length (−log₂ p) of the model's next-token
 * prediction.
 *
 * Start it from engine/:  uv run uvicorn main:app --port 5181
 */

export const ENGINE_URL: string =
  (import.meta.env.VITE_ENGINE_URL as string | undefined) ?? 'http://127.0.0.1:5181';

/** One vocabulary entry of a top-k readout. */
export interface TopTok {
  t: string;
  p: number;
}

/** What the engine knows about one servable model. `models` stays a plain
 * string list on the wire (the TUI reads it); `model_info` is the richer
 * per-model metadata newer engines add — merged here into one shape. */
export interface ModelInfo {
  name: string;
  kind: 'hub' | 'local';
  /** For local checkpoints: the note train.py saved alongside the weights. */
  note?: string;
  /** Positional budget (prompt + rollout tokens) — tiny for local models. */
  n_positions?: number;
  /** Sample prompts that actually fit this model's vocab/positions. */
  prompts?: string[];
}

export interface EngineHealth {
  device: string;
  models: ModelInfo[];
  default: string;
}

/** Only the /health fields this client reads — the wire also carries `ok` and
 * `loaded` (relayed whole by the MCP bridge, unused here). */
interface RawHealth {
  device: string;
  models: string[];
  default?: string;
  model_info?: Record<string, Omit<ModelInfo, 'name' | 'kind'> & { kind?: 'hub' | 'local' }>;
}

export interface LensResponse {
  model: string;
  prompt: string;
  /** Input tokens as display strings. */
  tokens: string[];
  /** Row names: ["embed", "layer 0", …, "final"]. */
  layers: string[];
  /** grid[layer][pos] = top-k classic-lens readout at that cell. */
  grid: TopTok[][][];
  /** −log₂ p(model's final top-1) per layer, at the prediction position. */
  bits: number[];
  /** Same, under the J-lens decode (null if jlens was off). */
  jbits: number[] | null;
  /** jtop[layer] = top-k of the J-decode at the prediction position. */
  jtop: TopTok[][] | null;
  /** The model's real next-token prediction. */
  pred: { token: string; p: number; bits: number };
  /** log₂(vocab) — the knows-nothing reference cost. */
  uniform: number;
  /** tokens[:n_prompt] are the prompt; the rest are server-side rollout. */
  n_prompt: number;
}

/** The depth ladder at one selected column — what /lens carries for the last
 * column, recomputed server-side for any other on selection. */
export interface ColumnResponse {
  model: string;
  prompt: string;
  pos: number;
  /** The model's real prediction *after* this column. */
  pred: { token: string; p: number; bits: number };
  bits: number[];
  jbits: number[] | null;
  jtop: TopTok[][] | null;
  uniform: number;
}

export async function fetchHealth(): Promise<EngineHealth> {
  const res = await fetch(`${ENGINE_URL}/health`);
  if (!res.ok) throw new Error(`engine ${res.status}`);
  const raw = (await res.json()) as RawHealth;
  return {
    device: raw.device,
    default: raw.default ?? raw.models[0] ?? 'gpt2',
    models: raw.models.map((name) => ({
      name,
      kind: name.startsWith('local/') ? 'local' : 'hub',
      ...(raw.model_info?.[name] ?? {})
    }))
  };
}

export async function fetchLens(req: {
  model: string;
  prompt: string;
  top_k?: number;
  jlens?: boolean;
  /** Greedy-decode up to this many tokens server-side; the lens covers prompt+continuation. */
  rollout?: number;
}): Promise<LensResponse> {
  const res = await fetch(`${ENGINE_URL}/lens`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ top_k: 5, jlens: true, rollout: 0, ...req })
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json();
}

/** /chat stream events: one "meta", then a "tok" per generated token, then "done". */
export interface ReasonMeta {
  event: 'meta';
  /** The templated prompt's tokens (chat markup included — honesty over tidiness). */
  tokens: string[];
  /** The prompt's token ids — prepend to the streamed ids for /column drill-ins. */
  ids: number[];
  layers: string[];
  uniform: number;
  n_prompt: number;
  temperature: number;
  /** The seed actually used (echoed even when auto-generated) — replay with it. */
  seed: number | null;
}
export interface ReasonTok {
  event: 'tok';
  /** Index of this token in the full (prompt + generated) sequence. */
  pos: number;
  /** Token id — hand back to /column via `ids` for fork-proof drill-ins. */
  id: number;
  t: string;
  /** The model's probability for this token when it emitted it. */
  p: number;
  /** −log₂ p_rung(this token): the classic-lens ladder of the column that produced it. */
  bits: number[];
  /** Each rung's own top-1 decode — the streamed grid column. */
  rtop: string[];
}
export interface ReasonDone {
  event: 'done';
  reason: 'eos' | 'budget';
}
export interface ReasonError {
  event: 'error';
  detail: string;
}
export type ReasonEvent = ReasonMeta | ReasonTok | ReasonDone | ReasonError;

/**
 * POST /chat and parse the SSE stream, invoking `onEvent` per event. Resolves
 * when the stream closes; `signal` aborts generation client-side.
 */
export async function streamChat(
  req: {
    model: string;
    prompt: string;
    chat?: boolean;
    thinking?: boolean;
    max_new?: number;
    /** 0 = greedy; Qwen recommends ~0.6 for thinking mode. Bits always price the true distribution. */
    temperature?: number;
    /** Omit for a fresh seed (echoed in meta); pass one to replay a run exactly. */
    seed?: number;
  },
  onEvent: (ev: ReasonEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${ENGINE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat: true, thinking: true, max_new: 512, ...req }),
    signal
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are blank-line separated; the tail may be a partial frame.
    const frames = buf.split('\n\n');
    buf = frames.pop() ?? '';
    for (const frame of frames) {
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as ReasonEvent);
      }
    }
  }
}

/** /train stream events — the Train·real lens's live feed. */
export interface TrainStart {
  event: 'start';
  params_m: number;
  device: string;
  epochs: number;
  n_train: number;
  n_test: number;
}
export interface TrainBatch {
  event: 'batch';
  epoch: number;
  /** Fraction of this epoch's batches done. */
  frac: number;
  loss: number;
}
export interface TrainEpoch {
  event: 'epoch';
  epoch: number;
  loss: number;
  /** Held-out exact-match over all 3 answer digits. */
  acc: number;
}
export interface TrainCheckpoint {
  event: 'checkpoint';
  /** Directory name — served by the engine as `local/<name>`. */
  name: string;
  note: string;
  /** Server-side filesystem path the checkpoint was written to. */
  path: string;
}
export interface TrainDone {
  event: 'done';
  acc: number;
  epochs_run: number;
}
export interface TrainError {
  event: 'error';
  detail: string;
}
export type TrainEvent =
  | TrainStart
  | TrainBatch
  | TrainEpoch
  | TrainCheckpoint
  | TrainDone
  | TrainError;

/**
 * POST /train and parse the SSE stream. One run at a time — the engine answers
 * 409 if a training run is already in progress. Aborting the fetch stops
 * training at the next batch; checkpoints already saved stay on disk.
 */
export async function streamTrain(
  req: { epochs?: number },
  onEvent: (ev: TrainEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${ENGINE_URL}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop() ?? '';
    for (const frame of frames) {
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as TrainEvent);
      }
    }
  }
}

/** One head's stats at a destination position (from /attn). The wire also
 * carries each head's `top` source list — read by the MCP bridge, not here. */
export interface AttnHead {
  layer: number;
  head: number;
  /** Entropy of the head's attention row, in bits — low = focused. */
  entropy: number;
  /** Mass on position 0 (the attention-sink no-op). */
  sink: number;
}

/** Only the /attn fields this client reads — the wire also carries `n_layers`
 * and the raw aggregate `agg` (both read by the MCP bridge). */
export interface AttnResponse {
  model: string;
  pos: number;
  seq: number;
  n_heads: number;
  /** Value-weighted aggregate (a·‖v‖ renormalized) — discounts sink stares. */
  vagg: number[];
  heads: AttnHead[];
  /** The requested layer/head's full value-weighted row, if asked for. */
  picked: { vrow: number[] } | null;
}

/** Where the computation that produced a token looked (destination = pos). */
export async function fetchAttn(req: {
  model: string;
  ids: number[];
  pos: number;
  layer?: number;
  head?: number;
}): Promise<AttnResponse> {
  const res = await fetch(`${ENGINE_URL}/attn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req)
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json();
}

export interface AblateResponse {
  model: string;
  pos: number;
  token: string;
  mask: [number, number];
  baseline: { p: number; bits: number; top: TopTok[] };
  masked: { p: number; bits: number; top: TopTok[] };
  /** What reading the masked region actually bought, in bits. */
  delta_bits: number;
}

/** Re-price ids[pos] with attention to [mask_start, mask_end) blocked for the
 * whole downstream computation — attention-as-bits. */
export async function fetchAblate(req: {
  model: string;
  ids: number[];
  pos: number;
  mask_start: number;
  mask_end: number;
  top_k?: number;
}): Promise<AblateResponse> {
  const res = await fetch(`${ENGINE_URL}/ablate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req)
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json();
}

export async function fetchColumn(req: {
  model: string;
  prompt: string;
  pos: number;
  top_k?: number;
  jlens?: boolean;
  rollout?: number;
  /** The exact sequence to lens (prompt ids + streamed ids): fork-proof, and
   * required for sampled traces. When set the server ignores prompt/rollout. */
  ids?: number[];
}): Promise<ColumnResponse> {
  const res = await fetch(`${ENGINE_URL}/column`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ top_k: 5, jlens: true, rollout: 0, ...req })
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json();
}
