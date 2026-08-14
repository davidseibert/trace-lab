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

export interface EngineHealth {
  ok: boolean;
  device: string;
  loaded: string[];
  models: string[];
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
  return res.json();
}

export async function fetchLens(req: {
  model: string;
  prompt: string;
  top_k?: number;
  jlens?: boolean;
  /** Greedy-decode up to this many tokens server-side; the lens covers prompt+continuation. */
  rollout?: number;
  /** Wrap the prompt in the model's chat template (thinking opens a <think> block). */
  chat?: boolean;
  thinking?: boolean;
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

export async function fetchColumn(req: {
  model: string;
  prompt: string;
  pos: number;
  top_k?: number;
  jlens?: boolean;
  rollout?: number;
  /** Must match the /lens or /chat request whose columns you're drilling into. */
  chat?: boolean;
  thinking?: boolean;
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
