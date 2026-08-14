// Shared scaffolding for the mini line-charts (cost, loss, bits, depth,
// train, token strip). This module owns the coordinate math every chart was
// repeating: the x/y scales, the M/L path string, and the click → index
// conversion. Each chart keeps its own frame size, series, colors, markers,
// and overlays — those are content, not scaffolding.

export interface ChartPad {
  l: number;
  r: number;
  t: number;
  b: number;
}

/** Default frame padding for the 520-wide step charts. */
export const DEFAULT_PAD: ChartPad = { l: 6, r: 6, t: 10, b: 16 };

export interface ChartScale {
  /** x for series index i — indices 0..n−1 span the plot width. */
  xAt: (i: number) => number;
  /** y for value v, clamped so v ≥ max sits on the top edge. */
  yAt: (v: number) => number;
  /** SVG "M x y L x y …" polyline through the series values. */
  path: (series: number[]) => string;
  /** clientX inside el → nearest series index, clamped to [0, n−1]. */
  indexAt: (clientX: number, el: Element) => number;
  /** Plot-area edges, for markers and guide lines that span the frame. */
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Builds the scale for a chart with n evenly spaced points where value max
 * maps to the top of the plot area. W/H are viewBox units; the rendered SVG
 * may stretch, which is why indexAt works in fractions of the on-screen width.
 */
export function chartScale(opts: {
  n: number;
  max: number;
  W: number;
  H: number;
  pad?: ChartPad;
}): ChartScale {
  const { n, max, W, H, pad = DEFAULT_PAD } = opts;
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  const xAt = (i: number) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * innerW);
  const yAt = (v: number) => H - pad.b - (Math.min(v, max) / max) * innerH;

  const path = (series: number[]) =>
    series
      .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`)
      .join(' ');

  const indexAt = (clientX: number, el: Element) => {
    const rect = el.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * W;
    const frac = (x - pad.l) / innerW;
    return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
  };

  return {
    xAt,
    yAt,
    path,
    indexAt,
    left: pad.l,
    right: W - pad.r,
    top: pad.t,
    bottom: H - pad.b
  };
}
