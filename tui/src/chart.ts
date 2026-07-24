/**
 * A braille line chart, as a pure function.
 *
 * Terminals have no pixels, but a braille cell has eight dots in a 2x4 grid, so
 * one character cell buys 2x horizontal and 4x vertical resolution. That is
 * enough to draw the depth curve — code length per rung — legibly in a corner
 * of the screen.
 *
 * No OpenTUI imports here on purpose: this is testable with plain `bun test`.
 */

/** Dot bit for subcolumn `cx` (0-1) and subrow `cy` (0-3) of a braille cell. */
const DOT = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
] as const;

const BRAILLE_BASE = 0x2800;

export interface ChartCell {
  char: string;
  fg: string;
  bg?: string;
}

export interface Series {
  values: number[];
  fg: string;
}

export interface ChartOptions {
  /** Size in character cells. */
  width: number;
  height: number;
  /** Drawn in order; a later series wins the colour of any cell they share. */
  series: Series[];
  /** Top of the y axis, in the same units as the values. */
  max: number;
  /** A horizontal dashed reference line (the uniform log₂V cost). */
  reference?: { value: number; fg: string };
  /** Index into the series to mark with a tinted column. */
  cursor?: number;
  /** Background tint for the cursor column. */
  cursorBg?: string;
}

/**
 * Rasterise the series into a grid of styled character cells.
 *
 * A braille cell carries one foreground colour, so where two curves cross, the
 * cell keeps every dot (the shapes stay readable) but takes the later series'
 * colour. With the classic lens drawn last, the curve you are reading as "the
 * code shortening with depth" is always the one that keeps its colour.
 */
export function renderChart(opts: ChartOptions): ChartCell[][] {
  const { width, height, series, max } = opts;
  if (width < 2 || height < 1) return [];

  const dotsW = width * 2;
  const dotsH = height * 4;
  // Per dot: which series (1-based) painted it. 0 = empty.
  const owner = new Uint8Array(dotsW * dotsH);

  const yAt = (v: number) => {
    const frac = max <= 0 ? 0 : Math.max(0, Math.min(1, v / max));
    return Math.round((1 - frac) * (dotsH - 1));
  };

  for (let s = 0; s < series.length; s++) {
    const values = series[s]!.values;
    const n = values.length;
    if (n === 0) continue;
    const xAt = (i: number) => (n === 1 ? 0 : Math.round((i / (n - 1)) * (dotsW - 1)));

    let px = xAt(0);
    let py = yAt(values[0]!);
    owner[py * dotsW + px] = s + 1;
    for (let i = 1; i < n; i++) {
      const x = xAt(i);
      const y = yAt(values[i]!);
      line(owner, dotsW, dotsH, px, py, x, y, s + 1);
      px = x;
      py = y;
    }
  }

  const cells: ChartCell[][] = [];
  const cursorX = opts.cursor === undefined ? -1 : cursorColumn(opts.cursor, series, width);
  const refRow =
    opts.reference === undefined ? -1 : Math.floor(yAt(opts.reference.value) / 4);

  for (let row = 0; row < height; row++) {
    const line: ChartCell[] = [];
    for (let col = 0; col < width; col++) {
      let bits = 0;
      let top = 0; // highest series index painting this cell
      for (let cy = 0; cy < 4; cy++) {
        for (let cx = 0; cx < 2; cx++) {
          const o = owner[(row * 4 + cy) * dotsW + (col * 2 + cx)]!;
          if (o !== 0) {
            bits |= DOT[cy]![cx]!;
            if (o > top) top = o;
          }
        }
      }
      const bg = col === cursorX ? opts.cursorBg : undefined;
      if (bits === 0) {
        // Nothing drawn here: the reference line shows through.
        if (row === refRow && opts.reference) {
          line.push({ char: col % 2 === 0 ? "┈" : " ", fg: opts.reference.fg, bg });
        } else {
          line.push({ char: " ", fg: "#000000", bg });
        }
        continue;
      }
      line.push({
        char: String.fromCharCode(BRAILLE_BASE + bits),
        fg: series[top - 1]!.fg,
        bg,
      });
    }
    cells.push(line);
  }
  return cells;
}

/** Which character column a rung index lands on. */
export function cursorColumn(index: number, series: Series[], width: number): number {
  const n = series[0]?.values.length ?? 0;
  if (n <= 1) return 0;
  const dot = Math.round((index / (n - 1)) * (width * 2 - 1));
  return Math.floor(dot / 2);
}

/** Bresenham, so consecutive rungs join up instead of floating as dots. */
function line(
  owner: Uint8Array,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  who: number,
): void {
  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  let x = x0;
  let y = y0;
  for (;;) {
    if (x >= 0 && x < w && y >= 0 && y < h) owner[y * w + x] = who;
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}
