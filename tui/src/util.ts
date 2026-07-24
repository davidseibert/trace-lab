/** Small pure helpers shared by the panels. */

/**
 * The window of indices to display when `total` items don't fit in `size`
 * slots, keeping `focus` visible with a little context either side.
 */
export function windowRange(total: number, focus: number, size: number): { start: number; end: number } {
  if (size >= total) return { start: 0, end: total };
  const half = Math.floor(size / 2);
  let start = Math.max(0, Math.min(focus - half, total - size));
  return { start, end: start + size };
}

/**
 * Hard-truncate a line to the panel width.
 *
 * Worth being strict about: a `<text>` wider than its box *wraps*, which costs
 * a row, overflows the panel, and silently draws over whatever is above it.
 */
export function clip(s: string, width: number): string {
  return s.length > width ? s.slice(0, Math.max(0, width)) : s;
}

/** Clamp to a fixed cell width, with an ellipsis when the token is too long. */
export function fit(s: string, width: number): string {
  if (width <= 0) return "";
  if (s.length === width) return s;
  if (s.length < width) return s.padEnd(width);
  return width === 1 ? "…" : s.slice(0, width - 1) + "…";
}

/** A left-aligned label, truncated hard (no ellipsis) — used for gutters. */
export function pad(s: string, width: number): string {
  return s.length > width ? s.slice(0, width) : s.padEnd(width);
}

/** `p` as a fixed-width percentage, e.g. " 97.3%". */
export function pct(p: number): string {
  return `${(p * 100).toFixed(1).padStart(5)}%`;
}

/** A bar of `width` cells filled proportionally to `frac`. */
export function bar(frac: number, width: number): string {
  const filled = Math.max(0, Math.min(width, Math.round(frac * width)));
  return "█".repeat(filled) + "░".repeat(width - filled);
}
