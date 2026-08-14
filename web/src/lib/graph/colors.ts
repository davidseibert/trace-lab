import { isBaseLabel, type GraphModel } from './graph';
import { goldenAngleHue, type SymColor } from '../mdl/palette';

/** Graph-lens name for the shared chip shape. */
export type LabelColor = SymColor;

/**
 * Stable, well-spread color per node label. Base labels are muted (they are the
 * raw data); learned substructures are vivid (they are the model). Hues march by
 * the golden angle so neighbouring ids stay maximally distinct — the same trick
 * the grammar lens uses for rule symbols. Both chip treatments are graph-specific
 * lightnesses; the type and the hue walk come from the shared palette.
 */
export function labelColor(m: GraphModel, id: number): LabelColor {
  const hue = goldenAngleHue(id);
  if (isBaseLabel(m, id)) {
    return {
      bg: `hsl(${hue} 16% 30%)`,
      fg: `hsl(${hue} 22% 88%)`,
      border: `hsl(${hue} 18% 44%)`
    };
  }
  return {
    bg: `hsl(${hue} 55% 30%)`,
    fg: `hsl(${hue} 90% 86%)`,
    border: `hsl(${hue} 65% 55%)`
  };
}
