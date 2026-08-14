import { describe, expect, test } from "bun:test";

import { cursorColumn, renderChart } from "./chart";
import { fit, windowRange } from "./util";

const DATA = "#ffb454";
const MODEL = "#5b9cff";

describe("renderChart", () => {
  test("returns a grid of the requested size", () => {
    const rows = renderChart({
      width: 20,
      height: 6,
      series: [{ values: [10, 8, 4, 2], fg: DATA }],
      max: 12,
    });
    expect(rows.length).toBe(6);
    expect(rows.every((r) => r.length === 20)).toBe(true);
  });

  test("a falling series starts high and ends low", () => {
    const rows = renderChart({
      width: 20,
      height: 8,
      series: [{ values: [16, 12, 8, 4, 1], fg: DATA }],
      max: 16,
    });
    const painted = (col: number) =>
      rows.map((r, i) => (r[col]!.char !== " " ? i : -1)).filter((i) => i >= 0);
    // Row 0 is the top of the chart, so a shorter code plots lower down.
    expect(Math.min(...painted(0))).toBeLessThan(Math.min(...painted(19)));
  });

  test("the later series wins the colour where curves overlap", () => {
    const flat = [5, 5, 5];
    const rows = renderChart({
      width: 10,
      height: 4,
      series: [
        { values: flat, fg: MODEL },
        { values: flat, fg: DATA },
      ],
      max: 10,
    });
    const painted = rows.flat().filter((c) => c.char !== " ");
    expect(painted.length).toBeGreaterThan(0);
    expect(painted.every((c) => c.fg === DATA)).toBe(true);
  });

  test("the cursor column is tinted", () => {
    const rows = renderChart({
      width: 12,
      height: 4,
      series: [{ values: [8, 6, 4, 2], fg: DATA }],
      max: 8,
      cursor: 0,
      cursorBg: "#333333",
    });
    expect(rows.every((r) => r[0]!.bg === "#333333")).toBe(true);
    expect(rows.every((r) => r[5]!.bg === undefined)).toBe(true);
  });

  test("the reference line only shows where nothing is drawn", () => {
    const rows = renderChart({
      width: 12,
      height: 4,
      series: [{ values: [8, 8, 8], fg: DATA }],
      max: 8,
      reference: { value: 4, fg: "#5c6478" },
    });
    expect(rows.flat().some((c) => c.char === "┈")).toBe(true);
  });

  test("cursorColumn spans the full width", () => {
    const s = [{ values: [1, 2, 3, 4, 5], fg: DATA }];
    expect(cursorColumn(0, s, 20)).toBe(0);
    expect(cursorColumn(4, s, 20)).toBe(19);
  });
});

describe("windowRange", () => {
  test("returns everything when it fits", () => {
    expect(windowRange(5, 2, 10)).toEqual({ start: 0, end: 5 });
  });

  test("keeps the focus inside the window", () => {
    for (let focus = 0; focus < 25; focus++) {
      const { start, end } = windowRange(25, focus, 8);
      expect(focus).toBeGreaterThanOrEqual(start);
      expect(focus).toBeLessThan(end);
      expect(end - start).toBe(8);
    }
  });
});

describe("fit", () => {
  test("pads short tokens and ellipsises long ones", () => {
    expect(fit("Paris", 8)).toBe("Paris   ");
    expect(fit("internationalisation", 8)).toBe("interna…");
  });
});
