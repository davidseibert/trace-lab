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
  /** Greedy-decode this many tokens server-side; the lens covers prompt+continuation. */
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

export async function fetchColumn(req: {
  model: string;
  prompt: string;
  pos: number;
  top_k?: number;
  jlens?: boolean;
  rollout?: number;
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
