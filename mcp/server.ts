/**
 * trace-lab MCP server — the bridge that lets an MCP client (Claude Code)
 * spectate the running TUI and query the lens engine directly.
 *
 * Two upstreams, both plain HTTP:
 *
 *   - the TUI's spectate sidecar (tui/src/spectate.ts, :5182) — read-only
 *     access to what the user currently sees: the literal character frame
 *     and the structured view state (model, prompt, selection, ladder).
 *   - the lens engine (engine/main.py, :5181) — the same /lens and /column
 *     endpoints the TUI calls, so the observer can run its own comparison
 *     prompts without touching the user's view.
 *
 * Registered in .mcp.json at the repo root; runs under bun over stdio.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const SPECTATE_URL = process.env.TUI_SPECTATE ?? "http://127.0.0.1:5182";
const ENGINE_URL = process.env.LENS_ENGINE ?? "http://127.0.0.1:5181";

/** One vocabulary entry of a top-k readout (mirrors tui/src/api.ts). */
interface TopTok {
  t: string;
  p: number;
}

interface LensResponse {
  model: string;
  prompt: string;
  tokens: string[];
  layers: string[];
  grid: TopTok[][][];
  bits: number[];
  jbits: number[] | null;
  jtop: TopTok[][] | null;
  pred: { token: string; p: number; bits: number };
  uniform: number;
  n_prompt: number;
}

/** The spectate sidecar's /state payload (shape defined in tui/src/App.tsx). */
interface SpectateState {
  view?: Record<string, unknown>;
  selection?: Record<string, unknown> | null;
  resp?: LensResponse | null;
  note?: string;
}

async function getJson<T>(url: string, hint: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  } catch {
    throw new Error(`unreachable: ${url} — ${hint}`);
  }
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function postJson<T>(url: string, body: unknown, hint: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // /lens on CPU with J-lens over a long prompt can genuinely take minutes.
      signal: AbortSignal.timeout(300_000),
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new Error(`timed out: ${url}`);
    }
    throw new Error(`unreachable: ${url} — ${hint}`);
  }
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j: any) => j.detail ?? res.statusText)
      .catch(() => res.statusText);
    throw new Error(`${url} → ${res.status}: ${String(detail)}`);
  }
  return (await res.json()) as T;
}

const TUI_HINT = "is the TUI running? Start it with `make tui` (or `cd tui && bun run start`)";
const ENGINE_HINT = "is the engine running? Start it with `make up` (or `cd engine && uv run uvicorn main:app --port 5181`)";

const fetchState = () => getJson<SpectateState>(`${SPECTATE_URL}/state`, TUI_HINT);

/** Round the float noise out of payloads we relay (p to 4 places, bits to 2). */
const round = (x: number, places: number) => {
  const f = 10 ** places;
  return Math.round(x * f) / f;
};
const trimTop = (tt: TopTok[], k: number) => tt.slice(0, k).map(({ t, p }) => ({ t, p: round(p, 4) }));

function ok(payload: unknown) {
  return {
    content: [
      {
        type: "text" as const,
        text: typeof payload === "string" ? payload : JSON.stringify(payload, null, 1),
      },
    ],
  };
}

function fail(e: unknown) {
  return {
    isError: true,
    content: [{ type: "text" as const, text: e instanceof Error ? e.message : String(e) }],
  };
}

const server = new McpServer({ name: "trace-lab", version: "0.1.0" });

server.registerTool(
  "get_screen",
  {
    title: "Capture the TUI screen",
    description:
      "The trace-lab TUI's current frame as plain text — the literal characters the user is looking " +
      "at right now (lens grid, depth chart, rung readout, status bar). Colors/shading are lost; use " +
      "get_state for the underlying numbers.",
    inputSchema: {},
  },
  async () => {
    try {
      const res = await fetch(`${SPECTATE_URL}/screen`, { signal: AbortSignal.timeout(10_000) }).catch(() => {
        throw new Error(`unreachable: ${SPECTATE_URL}/screen — ${TUI_HINT}`);
      });
      if (!res.ok) throw new Error(`${SPECTATE_URL}/screen → ${res.status} ${res.statusText}`);
      return ok(await res.text());
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "get_state",
  {
    title: "Read the TUI view state",
    description:
      "The structured state behind the user's current view: model, prompt, J-lens/rollout flags, the " +
      "selected cell (layer × token position), that cell's top-k readout, and the full depth ladder at " +
      "the selected column (bits and J-bits per rung, plus the J-decode top-k). Everything except the " +
      "full grid — fetch that with get_grid.",
    inputSchema: {},
  },
  async () => {
    try {
      const s = await fetchState();
      if (!s.view) return ok(s); // pre-first-publish note
      const r = s.resp;
      return ok({
        view: s.view,
        selection: s.selection ?? null,
        resp: r
          ? {
              model: r.model,
              prompt: r.prompt,
              tokens: r.tokens,
              n_prompt: r.n_prompt,
              layers: r.layers,
              pred: { ...r.pred, p: round(r.pred.p, 4), bits: round(r.pred.bits, 2) },
              uniform: round(r.uniform, 2),
              // Last-column ladder as returned by /lens (the selection's own
              // ladder, possibly for another column, is under selection.ladder).
              bits: r.bits.map((b) => round(b, 2)),
              jbits: r.jbits?.map((b) => round(b, 2)) ?? null,
            }
          : null,
      });
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "get_grid",
  {
    title: "Read the lens grid",
    description:
      "The full classic-lens grid behind the user's current view: for every layer × token position, " +
      "the top-k next-token readout. top_k=1 (default) matches what the grid cells display; raise it " +
      "for the runner-up tokens.",
    inputSchema: {
      top_k: z.number().int().min(1).max(8).default(1).describe("How many candidates per cell to include"),
    },
  },
  async ({ top_k }) => {
    try {
      const s = await fetchState();
      const r = s.resp;
      if (!r) return fail(new Error("the TUI has no lens response yet — the user hasn't run a prompt"));
      return ok({
        model: r.model,
        prompt: r.prompt,
        tokens: r.tokens,
        n_prompt: r.n_prompt,
        layers: r.layers,
        // grid[layer][pos] — same orientation as the TUI (rung 0 = embed).
        grid: r.grid.map((row) => row.map((cell) => trimTop(cell, top_k))),
      });
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "query_lens",
  {
    title: "Run the lens on a prompt",
    description:
      "Run the logit lens (and optionally the J-lens) on any prompt/model directly against the engine, " +
      "WITHOUT touching what the user sees in the TUI. Use this for comparison runs and hypothesis " +
      "checks. Returns the depth ladder in bits, the model's prediction, and the top-1 grid " +
      "(grid_top_k for more candidates). rollout > 0 greedily decodes up to that many tokens " +
      "server-side (stops at EOS) and lenses the whole thing. chat=true wraps the prompt in the " +
      "model's chat template; with thinking=true a Qwen3 reasoning model emits a <think> trace, so " +
      "chat+thinking+large rollout = lens over a full reasoning trace. Long rollouts return many " +
      "columns — consider jlens=false and grid_top_k=0 there and drill in with query_column.",
    inputSchema: {
      prompt: z.string().min(1),
      model: z.string().default("gpt2").describe("One of the engine's models — see engine_health"),
      jlens: z.boolean().default(true),
      rollout: z.number().int().min(0).max(1024).default(0),
      chat: z.boolean().default(false).describe("Wrap the prompt in the model's chat template"),
      thinking: z.boolean().default(true).describe("With chat: open a <think> block (Qwen3 reasoning)"),
      grid_top_k: z.number().int().min(0).max(8).default(1).describe("Candidates per grid cell; 0 omits the grid"),
    },
  },
  async ({ prompt, model, jlens, rollout, chat, thinking, grid_top_k }) => {
    try {
      const r = await postJson<LensResponse>(
        `${ENGINE_URL}/lens`,
        { prompt, model, jlens, rollout, chat, thinking, top_k: Math.max(1, grid_top_k) },
        ENGINE_HINT,
      );
      return ok({
        model: r.model,
        prompt: r.prompt,
        tokens: r.tokens,
        n_prompt: r.n_prompt,
        layers: r.layers,
        pred: { ...r.pred, p: round(r.pred.p, 4), bits: round(r.pred.bits, 2) },
        uniform: round(r.uniform, 2),
        bits: r.bits.map((b) => round(b, 2)),
        jbits: r.jbits?.map((b) => round(b, 2)) ?? null,
        jtop: r.jtop?.map((tt) => trimTop(tt, Math.max(1, grid_top_k))) ?? null,
        grid: grid_top_k > 0 ? r.grid.map((row) => row.map((cell) => trimTop(cell, grid_top_k))) : undefined,
      });
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "query_column",
  {
    title: "Ladder at one position",
    description:
      "The depth ladder (bits / J-bits per rung, J-decode top-k) at one token position of a prompt — " +
      "what the engine recomputes when the user selects a non-final column. pos indexes the prompt's " +
      "tokens (see query_lens's tokens array).",
    inputSchema: {
      prompt: z.string().min(1),
      pos: z.number().int().min(0),
      model: z.string().default("gpt2"),
      jlens: z.boolean().default(true),
      rollout: z.number().int().min(0).max(2048).default(0),
      chat: z.boolean().default(false).describe("Must match the query_lens call whose columns you're drilling into"),
      thinking: z.boolean().default(true),
      ids: z
        .array(z.number().int())
        .min(2)
        .max(4096)
        .optional()
        .describe(
          "Exact token ids to lens (fork-proof; required for sampled traces). When set, prompt/chat/rollout are ignored and the ladder measures the token that actually follows pos.",
        ),
      top_k: z.number().int().min(1).max(8).default(4),
    },
  },
  async ({ prompt, pos, model, jlens, rollout, chat, thinking, ids, top_k }) => {
    try {
      const c = await postJson<{
        model: string;
        prompt: string;
        pos: number;
        pred: { token: string; p: number; bits: number };
        bits: number[];
        jbits: number[] | null;
        jtop: TopTok[][] | null;
        uniform: number;
      }>(`${ENGINE_URL}/column`, { prompt, pos, model, jlens, rollout, chat, thinking, ids, top_k }, ENGINE_HINT);
      return ok({
        ...c,
        pred: { ...c.pred, p: round(c.pred.p, 4), bits: round(c.pred.bits, 2) },
        uniform: round(c.uniform, 2),
        bits: c.bits.map((b) => round(b, 2)),
        jbits: c.jbits?.map((b) => round(b, 2)) ?? null,
        jtop: c.jtop?.map((tt) => trimTop(tt, top_k)) ?? null,
      });
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "inspect_attention",
  {
    title: "Attention rows at one token",
    description:
      "Where the computation that produced a token looked: every head's attention row at destination " +
      "pos over the exact token sequence `ids` (use the /chat prompt ids + streamed ids; destination " +
      "for the token emitted at position P is P-1). Returns raw and value-weighted aggregates over all " +
      "heads (top sources only, to stay compact) plus the most focused heads. Pass layer+head for one " +
      "head's full row.",
    inputSchema: {
      model: z.string().default("Qwen/Qwen3-0.6B"),
      ids: z.array(z.number().int()).min(2).max(4096),
      pos: z.number().int().min(1).max(1500),
      layer: z.number().int().min(0).optional(),
      head: z.number().int().min(0).optional(),
      top: z.number().int().min(3).max(40).default(12).describe("How many aggregate sources / focused heads to return"),
    },
  },
  async ({ model, ids, pos, layer, head, top }) => {
    try {
      const a = await postJson<{
        seq: number; n_layers: number; n_heads: number;
        agg: number[]; vagg: number[];
        heads: { layer: number; head: number; entropy: number; sink: number; top: { pos: number; w: number; vw: number }[] }[];
        picked: { row: number[]; vrow: number[] } | null;
      }>(`${ENGINE_URL}/attn`, { model, ids, pos, layer, head }, ENGINE_HINT);
      const rank = (v: number[]) =>
        v.map((w, i) => ({ pos: i, w: round(w, 5) })).sort((x, y) => y.w - x.w).slice(0, top);
      return ok({
        pos, seq: a.seq, n_layers: a.n_layers, n_heads: a.n_heads,
        top_sources_raw: rank(a.agg),
        top_sources_value_weighted: rank(a.vagg),
        most_focused_heads: [...a.heads]
          .sort((x, y) => x.entropy - y.entropy)
          .slice(0, top)
          .map((h) => ({ ...h, top: h.top.slice(0, 5) })),
        picked: a.picked ? { vrow_top: rank(a.picked.vrow) } : null,
      });
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "ablate_region",
  {
    title: "Re-price a token without a region",
    description:
      "Attention-as-bits, causally: forbid the whole computation after mask_end from attending to " +
      "sources [mask_start, mask_end), then re-price ids[pos]. delta_bits is what reading that region " +
      "actually bought (e.g. mask the <think> span and re-price the answer token). Same ids contract " +
      "as query_column/inspect_attention.",
    inputSchema: {
      model: z.string().default("Qwen/Qwen3-0.6B"),
      ids: z.array(z.number().int()).min(2).max(4096),
      pos: z.number().int().min(1),
      mask_start: z.number().int().min(0),
      mask_end: z.number().int().min(1),
      top_k: z.number().int().min(1).max(20).default(5),
    },
  },
  async ({ model, ids, pos, mask_start, mask_end, top_k }) => {
    try {
      return ok(await postJson(`${ENGINE_URL}/ablate`, { model, ids, pos, mask_start, mask_end, top_k }, ENGINE_HINT));
    } catch (e) {
      return fail(e);
    }
  },
);

server.registerTool(
  "engine_health",
  {
    title: "Engine health",
    description:
      "Ping the lens engine: device (cuda/cpu), which models are available, which are already resident, " +
      "and the default. Also reports whether the TUI spectate sidecar is reachable.",
    inputSchema: {},
  },
  async () => {
    const [engine, tui] = await Promise.all([
      getJson(`${ENGINE_URL}/health`, ENGINE_HINT).then(
        (h) => h,
        (e) => ({ error: e instanceof Error ? e.message : String(e) }),
      ),
      fetch(`${SPECTATE_URL}/state`, { signal: AbortSignal.timeout(3_000) }).then(
        (r) => r.ok,
        () => false,
      ),
    ]);
    return ok({ engine, tui_spectate: tui ? "reachable" : `unreachable at ${SPECTATE_URL} — ${TUI_HINT}` });
  },
);

await server.connect(new StdioServerTransport());
