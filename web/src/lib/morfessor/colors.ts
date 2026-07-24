export interface SymColor {
  bg: string;
  fg: string;
  border: string;
}

const NEUTRAL: SymColor = {
  bg: 'hsl(222 12% 24%)',
  fg: 'hsl(222 18% 82%)',
  border: 'hsl(222 12% 34%)'
};

/**
 * Stable color per morph STRING (the Morfessor lens has no integer ids — morphs
 * are their own surface forms). Single characters stay neutral, so an unsplit
 * letter reads as "atomic"; multi-char morphs get a vivid, stable hue.
 */
export function morphColor(m: string): SymColor {
  if (m.length <= 1) return NEUTRAL;
  let h = 0;
  for (const ch of m) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const hue = h % 360;
  return {
    bg: `hsl(${hue} 55% 26%)`,
    fg: `hsl(${hue} 90% 84%)`,
    border: `hsl(${hue} 60% 46%)`
  };
}
