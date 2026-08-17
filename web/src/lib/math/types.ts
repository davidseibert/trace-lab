export type MathLabId = 'probability-bits' | 'entropy' | 'arithmetic' | 'cross-entropy';

export type LessonBlock =
  | { kind: 'text'; title?: string; text: string }
  | { kind: 'equation'; title?: string; lines: string[]; note?: string }
  | { kind: 'list'; title?: string; items: string[] }
  | { kind: 'callout'; title: string; text: string; tone?: 'plain' | 'good' | 'warn' };

export interface MathTerm {
  symbol: string;
  meaning: string;
  unit?: string;
}

export interface MathLesson {
  id: string;
  chapter: string;
  title: string;
  shortTitle: string;
  question: string;
  summary: string;
  prerequisites: string[];
  terms: MathTerm[];
  blocks: LessonBlock[];
  lab: MathLabId;
  seenIn: { lens: string; label: string; note: string }[];
}
