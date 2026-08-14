/**
 * Spectate sidecar: a tiny HTTP server inside the TUI process so an outside
 * observer (the trace-lab MCP server, `mcp/server.ts`) can see what the user
 * sees — both the literal character frame and the structured view state.
 *
 * Read-only by design: it exposes state, it never mutates it. Two endpoints:
 *
 *   GET /state   the app state last published by App.tsx (JSON)
 *   GET /screen  the current rendered frame as plain text, straight off the
 *                renderer's character buffer — exactly what's on the terminal
 *
 * Best-effort by design, too: if the port is taken (say, two TUIs at once),
 * the sidecar just doesn't start; the TUI itself must never die for it.
 *
 * TUI_SPECTATE_PORT=0 disables it. TUI_SPECTATE_HOST defaults to loopback;
 * the compose file sets 0.0.0.0 so the mapped container port is reachable.
 */
import type { CliRenderer } from "@opentui/core";

let renderer: CliRenderer | null = null;
let state: unknown = null;

const decoder = new TextDecoder();

/** App.tsx calls this on every state change; /state serves the latest. */
export function publishState(next: unknown): void {
  state = next;
}

/** Start the sidecar (idempotent per process — call once from index.tsx). */
export function attachSpectate(r: CliRenderer): void {
  renderer = r;
  const port = Number(process.env.TUI_SPECTATE_PORT ?? 5182);
  if (!port) return;
  try {
    Bun.serve({
      port,
      hostname: process.env.TUI_SPECTATE_HOST ?? "127.0.0.1",
      fetch(req) {
        const path = new URL(req.url).pathname;
        if (path === "/state") {
          return Response.json(state ?? { note: "TUI is up but has not published state yet" });
        }
        if (path === "/screen") {
          if (!renderer || renderer.isDestroyed) {
            return new Response("renderer gone", { status: 503 });
          }
          const frame = decoder.decode(renderer.currentRenderBuffer.getRealCharBytes(true));
          return new Response(frame, {
            headers: { "content-type": "text/plain; charset=utf-8" },
          });
        }
        return new Response("not found", { status: 404 });
      },
    });
  } catch {
    // Port busy or bind refused: spectating is optional, the TUI is not.
  }
}
