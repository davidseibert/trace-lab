/** Stable, well-spread colour per symbol id (golden-angle hues), matching the
 *  palette idiom used by the other lenses' colour helpers. */

import { goldenAngleHue, type SymColor } from '../mdl/palette';

export type { SymColor };

export function symColorFor(id: number): SymColor {
  const hue = goldenAngleHue(id);
  // Coder-specific lightnesses (slightly brighter than the shared vivid chip);
  // the type and the hue walk come from the shared palette.
  return {
    bg: `hsl(${hue} 55% 30%)`,
    fg: `hsl(${hue} 90% 86%)`,
    border: `hsl(${hue} 60% 50%)`
  };
}
