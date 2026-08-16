/**
 * A hand-authored 5×7 pixel font for the Hopfield lens's stored patterns.
 * Glyphs are the pedagogically right patterns: you can *see* retrieval succeed
 * (the corrupted A snaps back to an A) and *see* metastable averaging (the
 * state becomes a gray ghost of several letters at once).
 *
 * The set deliberately includes correlated pairs (B/P, E/L, H/U share strokes)
 * — the capacity panel needs crosstalk to demonstrate, and the separation
 * statistic Δ needs something to be small about.
 */

export const GLYPH_ROWS = 7;
export const GLYPH_COLS = 5;
export const GLYPH_DIM = GLYPH_ROWS * GLYPH_COLS; // d = 35

// 7 strings of 5 chars per glyph: '#' = +1 (ink), '.' = −1 (paper).
export const FONT: Record<string, string[]> = {
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  B: ['####.', '#...#', '#...#', '####.', '#...#', '#...#', '####.'],
  C: ['.###.', '#...#', '#....', '#....', '#....', '#...#', '.###.'],
  E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
  H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  J: ['..###', '...#.', '...#.', '...#.', '...#.', '#..#.', '.##..'],
  L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
  P: ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
  S: ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
  T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
  U: ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
  X: ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#']
};

export const GLYPH_CHARS = Object.keys(FONT);

/** One glyph as a flat ±1 vector, row-major over the 7×5 bitmap. */
export function glyphVector(ch: string): Float64Array {
  const rows = FONT[ch];
  if (!rows) throw new Error(`no glyph for ${JSON.stringify(ch)}`);
  const v = new Float64Array(GLYPH_DIM);
  for (let r = 0; r < GLYPH_ROWS; r++)
    for (let c = 0; c < GLYPH_COLS; c++) v[r * GLYPH_COLS + c] = rows[r][c] === '#' ? 1 : -1;
  return v;
}

/** Keep only characters the font knows, uppercased, deduplicated, in order. */
export function validGlyphChars(s: string): string[] {
  const out: string[] = [];
  for (const ch of s.toUpperCase()) if (FONT[ch] && !out.includes(ch)) out.push(ch);
  return out;
}
