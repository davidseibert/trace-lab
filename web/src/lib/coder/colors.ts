/** Stable, well-spread colour per symbol id (golden-angle hues), matching the
 *  palette idiom used by the other lenses' colour helpers. */

export interface SymColor {
  bg: string;
  fg: string;
  border: string;
}

export function symColorFor(id: number): SymColor {
  const hue = (id * 137.508) % 360; // golden angle -> maximally distinct hues
  return {
    bg: `hsl(${hue} 55% 30%)`,
    fg: `hsl(${hue} 90% 86%)`,
    border: `hsl(${hue} 60% 50%)`
  };
}
