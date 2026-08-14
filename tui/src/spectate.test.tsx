/**
 * Renders the app headlessly with the spectate sidecar attached and asserts
 * that an outside observer (the MCP bridge, mcp/server.ts) sees what the user
 * sees: the exact character frame and the structured view state.
 *
 * Like app.test.tsx: skipped unless an engine is reachable.
 */
import { createTestRenderer } from "@opentui/core/testing";
import { createRoot } from "@opentui/react";
import { beforeAll, describe, expect, test } from "bun:test";

import { App } from "./App";
import { ENGINE_URL, fetchHealth } from "./api";
import { attachSpectate } from "./spectate";

const PORT = 5198; // off the real 5182 so a live TUI never collides with tests

let online = false;

beforeAll(async () => {
  online = await fetchHealth().then(
    () => true,
    () => false,
  );
  if (!online) console.warn(`no engine at ${ENGINE_URL} — skipping spectate test`);
});

describe("spectate sidecar", () => {
  test("serves the frame and the view state over HTTP", async () => {
    if (!online) return;

    process.env.TUI_SPECTATE_PORT = String(PORT);
    const { renderer, renderOnce, captureCharFrame } = await createTestRenderer({
      width: 140,
      height: 40,
    });
    attachSpectate(renderer);
    createRoot(renderer).render(<App />);

    // Poll until the default prompt's lens lands (network + maybe weights).
    let frame = "";
    const deadline = Date.now() + 150_000;
    while (Date.now() < deadline) {
      await renderOnce();
      frame = captureCharFrame();
      if (frame.includes("Paris")) break;
      await Bun.sleep(250);
    }
    expect(frame).toContain("Paris");
    await renderOnce(); // let the publish effect flush the latest state

    // /screen serves the same characters the user's terminal shows.
    const screen = await fetch(`http://127.0.0.1:${PORT}/screen`).then((r) => r.text());
    expect(screen).toContain("lens grid");
    expect(screen).toContain("Paris");

    // /state serves the structured view behind it.
    const state = (await fetch(`http://127.0.0.1:${PORT}/state`).then((r) => r.json())) as any;
    expect(state.view.model).toBe("gpt2");
    expect(state.view.prompt).toContain("Eiffel");
    expect(state.selection.layer).toBe("final");
    expect(state.selection.ladder.bits.length).toBe(state.resp.layers.length);
    expect(state.resp.grid.length).toBe(state.resp.layers.length);

    renderer.destroy();
  }, 180_000);
});
