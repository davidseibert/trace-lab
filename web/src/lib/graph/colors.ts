import { isBaseLabel, type GraphModel } from './graph';

export interface LabelColor {
  bg: string;
  fg: string;
  border: string;
}

/**
 * Stable, well-spread color per node label. Base labels are muted (they are the
 * raw data); learned substructures are vivid (they are the model). Hues march by
 * the golden angle so neighbouring ids stay maximally distinct — the same trick
 * the grammar lens uses for rule symbols.
 */
export function labelColor(m: GraphModel, id: number): LabelColor {
  const hue = (id * 137.508) % 360;
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
