/**
 * The palette matches the web front-end's, on purpose: the two clients show the
 * same lens and should read the same. The two load-bearing colours are the MDL currency's
 * two halves — gold is L(D|M) (the data given the model: the classic lens's
 * per-layer code length) and blue is L(M) (the model itself: the J-lens, which
 * asks what the *remaining computation* makes of a rung).
 */
export const theme = {
  bg: "#0e1014",
  panel: "#171b23",
  border: "#272d3a",
  border2: "#333b4d",
  text: "#e7ebf3",
  muted: "#8b93a7",
  faint: "#5c6478",
  data: "#ffb454", // classic logit lens
  model: "#5b9cff", // J-lens
  ok: "#4dc07d",
  bad: "#ff6b6b",
  /** Legible foreground on a saturated `data`-coloured cell. */
  onData: "#06121f",
} as const;

function channels(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function toHex(r: number, g: number, b: number): string {
  const h = (v: number) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

/** Linear blend between two hex colours; `t` in [0, 1]. */
export function mix(from: string, to: string, t: number): string {
  const a = channels(from);
  const b = channels(to);
  const k = Math.max(0, Math.min(1, t));
  return toHex(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k);
}

/**
 * Cell shading for the lens grid: dark background -> warm gold as the layer
 * grows confident. Same ramp the Textual TUI used, now that we know the gold
 * it was reaching for is exactly the web front-end's L(D|M) colour.
 */
export function probColors(p: number): { bg: string; fg: string } {
  return {
    bg: mix(theme.bg, theme.data, p),
    fg: p > 0.55 ? theme.onData : theme.text,
  };
}
