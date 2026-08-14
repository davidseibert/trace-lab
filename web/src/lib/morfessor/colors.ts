import { NEUTRAL, vividColor, type SymColor } from '../mdl/palette';

export type { SymColor };

/**
 * Stable color per morph STRING (the Morfessor lens has no integer ids — morphs
 * are their own surface forms). Single characters stay neutral, so an unsplit
 * letter reads as "atomic"; multi-char morphs get a vivid, stable hue — hashed
 * from the string, since there is no id for the shared golden-angle walk.
 */
export function morphColor(m: string): SymColor {
  if (m.length <= 1) return NEUTRAL;
  let h = 0;
  for (const ch of m) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return vividColor(h % 360);
}
