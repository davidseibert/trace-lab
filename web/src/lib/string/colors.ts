import { isTerminal, ruleIndexOf, type GrammarModel } from './grammar';
import { NEUTRAL, goldenAngleHue, vividColor, type SymColor } from '../mdl/palette';

export type { SymColor };

/** Stable, well-spread color per symbol. Terminals neutral, rules vivid. */
export function symColor(m: GrammarModel, id: number): SymColor {
  if (isTerminal(m, id)) return NEUTRAL;
  return vividColor(goldenAngleHue(ruleIndexOf(m, id)));
}
