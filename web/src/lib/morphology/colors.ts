import { isTerminal, ruleIndexOf, type MorphModel } from './morphology';
import { NEUTRAL, goldenAngleHue, vividColor, type SymColor } from '../mdl/palette';

export type { SymColor };

/** Stable, well-spread color per morph. Base chars neutral, learned morphs vivid. */
export function symColor(m: MorphModel, id: number): SymColor {
  if (isTerminal(m, id)) return NEUTRAL;
  return vividColor(goldenAngleHue(ruleIndexOf(m, id)));
}
