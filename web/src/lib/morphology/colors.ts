import { isTerminal, ruleIndexOf, type MorphModel } from './morphology';

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

/** Stable, well-spread color per morph. Base chars neutral, learned morphs vivid. */
export function symColor(m: MorphModel, id: number): SymColor {
  if (isTerminal(m, id)) return NEUTRAL;
  const idx = ruleIndexOf(m, id);
  const hue = (idx * 137.508) % 360; // golden angle -> maximally distinct hues
  return {
    bg: `hsl(${hue} 55% 26%)`,
    fg: `hsl(${hue} 90% 84%)`,
    border: `hsl(${hue} 60% 46%)`
  };
}
