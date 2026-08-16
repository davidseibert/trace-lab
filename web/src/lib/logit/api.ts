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

/** POST a JSON body to an engine endpoint. On a non-ok reply, surfaces the
 * FastAPI error `detail` (falling back to the status text) as the thrown
 * Error's message. */
async function post(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  const res = await fetch(`${ENGINE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail ?? res.statusText).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res;
}

/** Yield the parsed JSON payload of each `data:` line in an SSE response body.
 * Frames are blank-line separated; the buffer's tail may be a partial frame. */
async function* sseEvents<T>(res: Response): AsyncGenerator<T> {
  if (!res.body) throw new Error(res.statusText);
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
        if (line.startsWith('data: ')) yield JSON.parse(line.slice(6)) as T;
      }
    }
  }
}

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
  /** Wrap the prompt in the model's chat template (server-side render). */
  chat?: boolean;
  /** With chat: open a <think> block (Qwen3 reasoning). */
  thinking?: boolean;
}): Promise<LensResponse> {
  const res = await post('/lens', { top_k: 5, jlens: true, rollout: 0, ...req });
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
  const res = await post('/chat', { chat: true, thinking: true, max_new: 512, ...req }, signal);
  for await (const ev of sseEvents<ReasonEvent>(res)) onEvent(ev);
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
  const res = await post('/train', req, signal);
  for await (const ev of sseEvents<TrainEvent>(res)) onEvent(ev);
}

/** One head's stats at a destination position (from /attn). */
export interface AttnHead {
  layer: number;
  head: number;
  /** Entropy of the head's attention row, in bits — low = focused. */
  entropy: number;
  /** Mass on position 0 (the attention-sink no-op). */
  sink: number;
  /** The head's strongest sources: raw weight `w` and value-weighted `vw`. */
  top: { pos: number; w: number; vw: number }[];
}

/** The full /attn wire contract — canonical for every consumer (this client
 * and the MCP bridge both import it; each reads the fields it needs). */
export interface AttnResponse {
  model: string;
  pos: number;
  seq: number;
  n_layers: number;
  n_heads: number;
  /** Raw aggregate: mean attention received per source over all layers×heads. */
  agg: number[];
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
  const res = await post('/attn', req);
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
  const res = await post('/ablate', req);
  return res.json();
}

/** Retrieval regimes — same thresholds engine-side as the toy lens's exported
 * constants in lib/hopfield/hopfield.ts, so both lenses speak one vocabulary. */
export type HopfieldRegime = 'retrieval' | 'metastable' | 'global';

/** One head's behavior at one inverse-temperature multiple γ (γ = 1 is the
 * model's own 1/√d_k). */
export interface HopfieldCurvePoint {
  gamma: number;
  /** Entropy of the γ-rescaled attention row, in bits. */
  entropy_bits: number;
  max_w: number;
  /** 2^entropy — effective number of mixed source positions. */
  eff_k: number;
  regime: HopfieldRegime;
}

export interface HopfieldHead {
  layer: number;
  head: number;
  curves: HopfieldCurvePoint[];
  /** The regime at γ = 1 (null if 1.0 wasn't in the sweep). */
  regime_at_1: HopfieldRegime | null;
}

/** The full /hopfield wire contract — canonical for this client and the MCP
 * bridge. Every attention head read as one-step modern-Hopfield retrieval. */
export interface HopfieldHeadsResponse {
  model: string;
  pos: number;
  seq: number;
  n_layers: number;
  n_heads: number;
  gammas: number[];
  /** The forwarded tokens (ids[:pos+1]) as display strings. */
  tokens: string[];
  heads: HopfieldHead[];
}

/** Read every head at one destination as Hopfield retrieval across a γ sweep. */
export async function fetchHopfield(req: {
  model: string;
  prompt: string;
  /** Destination position; -1 (default) = the last token. */
  pos?: number;
  gammas?: number[];
}): Promise<HopfieldHeadsResponse> {
  const res = await post('/hopfield', req);
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
  const res = await post('/column', { top_k: 5, jlens: true, rollout: 0, ...req });
  return res.json();
}
