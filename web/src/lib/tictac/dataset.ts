/**
 * The Tic·tac lens's Dataset builder — lens-local on purpose (never registered
 * in DATASETS, so it doesn't appear in Mini·GPT's picker). trainTrace builds
 * the ModelConfig straight from these fields: 1 block, ~5.7k params, no
 * model.ts changes.
 *
 * Token scheme: '·' is a start token (so the FIRST move is a predictable
 * next-token like every other), then cells '0'..'8'. A full game is at most
 * 10 tokens; the shifted input is at most 9 ≤ ctxLen. The always-causal
 * attention mask is exactly right — a game is a causal sequence.
 */

import type { Dataset } from '../llm/datasets';
import { generateCorpus, type CorpusKind } from './game';

export const TIC_VOCAB = ['·', '0', '1', '2', '3', '4', '5', '6', '7', '8'];

/** Token colors by D₄ orbit class: corners --model blue, edges --data orange,
 * center --chosen yellow, '·' neutral grey — so every token-colored view in
 * the shared components carries the symmetry story for free. */
export const TIC_COLORS: Record<string, string> = {
  '·': '#8b93a7',
  '0': '#5b9cff', '2': '#5b9cff', '6': '#5b9cff', '8': '#5b9cff',
  '1': '#ffb454', '3': '#ffb454', '5': '#ffb454', '7': '#ffb454',
  '4': '#ffd166'
};

/** Orbit class of a cell, for scatter/labels. */
export const orbitClass = (cell: number): 'corner' | 'edge' | 'center' =>
  cell === 4 ? 'center' : cell % 2 === 0 ? 'corner' : 'edge';

export function tictacDataset(kind: CorpusKind, nGames: number, seed: number): Dataset {
  return {
    key: 'tictac',
    label: 'Tic·tac',
    subtitle: `${kind} · ${nGames} games`,
    vocab: TIC_VOCAB,
    tokenColors: TIC_COLORS,
    trainData: generateCorpus(kind, nGames, seed),
    // The forced block: X holds 0 and 1, O must play 2 — a probe with a UNIQUE
    // optimal answer, so the trace's built-in targetProb reads as "probability
    // on the forced block".
    probe: ['·', '0', '4', '1'],
    probeTarget: '2',
    embDim: 24,
    ctxLen: 10,
    nHeads: 2,
    ffnHid: 48,
    lr: 0.01,
    queryPlaceholder: '· 0 4 1',
    stageDemo: {} // Tic·tac has no Explain panel; field required by the interface
  };
}
