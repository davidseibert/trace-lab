/**
 * Client for the engine service (engine/main.py — FastAPI over torch).
 *
 * The lens is Python and this TUI is Bun, so they meet over HTTP. The engine
 * keeps models resident, which is why relaunching the TUI is instant and
 * switching models mid-session costs nothing after the first request.
 *
 * The Svelte web front-end speaks the same payload against the same engine —
 * two front-ends, one service.
 */

/** Compose injects this; falls back to a locally-run `uvicorn main:app`. */
export const ENGINE_URL = process.env.LENS_ENGINE ?? "http://127.0.0.1:5181";

/** The wire types are shared with the web client — web/src/lib/logit/api.ts
 * is canonical (type-only import, erased at runtime under bun). */
export type { TopTok, LensResponse, ColumnResponse } from "../../web/src/lib/logit/api";
import type { LensResponse, ColumnResponse } from "../../web/src/lib/logit/api";

export interface EngineHealth {
  ok: boolean;
  device: string;
  loaded: string[];
  models: string[];
  default: string;
}

export async function fetchHealth(): Promise<EngineHealth> {
  const res = await fetch(`${ENGINE_URL}/health`);
  if (!res.ok) throw new Error(`engine ${res.status}`);
  return res.json() as Promise<EngineHealth>;
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
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 8, jlens: true, rollout: 0, ...req }),
  });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j: any) => j.detail ?? res.statusText)
      .catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json() as Promise<LensResponse>;
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
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 8, jlens: true, rollout: 0, ...req }),
  });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j: any) => j.detail ?? res.statusText)
      .catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json() as Promise<ColumnResponse>;
}
