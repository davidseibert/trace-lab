/**
 * Renders the app against a real engine, headlessly, and asserts the panels
 * actually contain the lens.
 *
 * Skipped unless an engine is reachable, so `bun test` stays useful without a
 * GPU: start one with `make engine` and re-run.
 */
import { createTestRenderer } from "@opentui/core/testing";
import { createRoot } from "@opentui/react";
import { beforeAll, describe, expect, test } from "bun:test";

import { App } from "./App";
import { ENGINE_URL, fetchHealth } from "./api";

let online = false;

beforeAll(async () => {
  online = await fetchHealth().then(
    () => true,
    () => false,
  );
  if (!online) console.warn(`no engine at ${ENGINE_URL} — skipping render test`);
});

describe("App", () => {
  test("renders the lens for the default prompt", async () => {
    if (!online) return;

    const { renderer, renderOnce, captureCharFrame } = await createTestRenderer({
      width: 140,
      height: 40,
    });
    createRoot(renderer).render(<App />);

    // `waitForFrame` counts render passes, not seconds — and what we're waiting
    // on is a network round trip that may include loading weights. Poll.
    let frame = "";
    const deadline = Date.now() + 150_000;
    while (Date.now() < deadline) {
      await renderOnce();
      frame = captureCharFrame();
      if (frame.includes("Paris")) break;
      await Bun.sleep(250);
    }

    expect(frame).toContain("lens grid"); // panel titles
    expect(frame).toContain("code length by depth");
    expect(frame).toContain("rung readout");
    expect(frame).toContain("embed"); // the bottom rung
    expect(frame).toContain("final"); // and the top one
    expect(frame).toContain("uniform"); // the log₂V reference
    expect(frame).toContain("J-lens");
    expect(frame).toMatch(/\d+\.\d+b/); // a code length, in bits

    renderer.destroy();
  }, 180_000);
});
