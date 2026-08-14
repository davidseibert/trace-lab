/**
 * Shared palette idiom for the lens views.
 *
 * Every lens colours structure the same way: atomic/base symbols stay NEUTRAL
 * (they are the raw data), learned symbols get a vivid, stable hue (they are
 * the model). Hues march by the golden angle so consecutive ids stay
 * maximally distinct. A lens that needs a different chip treatment (the graph
 * lens's muted base labels, the coder's slightly brighter chips) keeps its own
 * hsl lines but shares the type, the neutral, and the hue walk.
 */

export interface SymColor {
  bg: string;
  fg: string;
  border: string;
}

/** Neutral chip for atomic/base symbols — the raw data, not the model. */
export const NEUTRAL: SymColor = {
  bg: 'hsl(222 12% 24%)',
  fg: 'hsl(222 18% 82%)',
  border: 'hsl(222 12% 34%)'
};

/** Golden-angle hue walk: consecutive ids get maximally distinct hues. */
export const goldenAngleHue = (id: number): number => (id * 137.508) % 360;

/** The standard vivid chip for a learned symbol at a given hue (used by the
 *  string, morphology, and morfessor lenses). */
export const vividColor = (hue: number): SymColor => ({
  bg: `hsl(${hue} 55% 26%)`,
  fg: `hsl(${hue} 90% 84%)`,
  border: `hsl(${hue} 60% 46%)`
});
