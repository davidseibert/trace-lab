/**
 * Toy training corpora for the transformer lens. Each is small enough to learn
 * in a few hundred steps yet rich enough to show structure forming: the model
 * has to route information through attention to mix two colors correctly.
 *
 * `probe` is the fixed input we visualise every step (independent of the random
 * training sample), so the panels show one sentence's internals sharpening as
 * loss falls — e.g. watching "red + blue =" converge on "purple".
 */

export interface Dataset {
  key: string;
  label: string;
  subtitle: string;
  /** Ordered vocabulary; index === token id. */
  vocab: string[];
  /** Display color per token. */
  tokenColors: Record<string, string>;
  /** Full sequences; the model predicts each next token. */
  trainData: string[][];
  /** The fixed input we run + visualise every step (model predicts what follows). */
  probe: string[];
  /** What the probe *should* learn to predict next, for the UI to celebrate. */
  probeTarget: string;
  embDim: number;
  ctxLen: number;
  nHeads: number;
  ffnHid: number;
  lr: number;
  queryPlaceholder: string;
  /** Per-stage plain-language copy for the Explain panel (Phase 3). */
  stageDemo: Record<string, string>;
}

export const DATASETS: Record<string, Dataset> = {
  'color-arithmetic': {
    key: 'color-arithmetic',
    label: 'Color Arithmetic',
    subtitle: 'A tiny transformer learns to mix colors',
    vocab: ['red', 'blue', 'yellow', 'green', 'orange', 'purple', '+', '='],
    tokenColors: {
      red: '#ff3355',
      blue: '#3388ff',
      yellow: '#ffcc00',
      green: '#00dd66',
      orange: '#ff7722',
      purple: '#bb44ff',
      '+': '#667788',
      '=': '#667788'
    },
    trainData: [
      ['red', '+', 'blue', '=', 'purple'],
      ['blue', '+', 'red', '=', 'purple'],
      ['red', '+', 'yellow', '=', 'orange'],
      ['yellow', '+', 'red', '=', 'orange'],
      ['blue', '+', 'yellow', '=', 'green'],
      ['yellow', '+', 'blue', '=', 'green']
    ],
    probe: ['red', '+', 'blue', '='],
    probeTarget: 'purple',
    embDim: 8,
    ctxLen: 5,
    nHeads: 2,
    ffnHid: 16,
    lr: 0.005,
    queryPlaceholder: 'red + blue =',
    stageDemo: {
      input: 'For "red + blue =", each token becomes one row.',
      embedding: 'The bars show each token fingerprint as positive/negative values.',
      posenc: 'Same token in a different spot gets a slightly different pattern.',
      attention: 'Heatmap cells show how strongly one token looks at another.',
      ffn: 'Hidden bars fire differently for each training example.',
      output: 'Taller bars mean a stronger guess for that token.'
    }
  },
  'pablo-painting': {
    key: 'pablo-painting',
    label: "Pablo's Paintings",
    subtitle: 'A tiny transformer learns how paint mixes on colored paper',
    vocab: [
      'pablo',
      'painted',
      'the',
      'red',
      'paper',
      'yellow',
      '.',
      'blue',
      'orange',
      'green',
      'purple',
      'before',
      ',',
      'it',
      'was',
      'is',
      'now'
    ],
    tokenColors: {
      pablo: '#e8a838',
      painted: '#c4956a',
      the: '#778899',
      paper: '#d4c5a9',
      before: '#8899aa',
      it: '#778899',
      was: '#778899',
      is: '#778899',
      now: '#778899',
      red: '#ff3355',
      blue: '#3388ff',
      yellow: '#ffcc00',
      green: '#00dd66',
      orange: '#ff7722',
      purple: '#bb44ff',
      '.': '#556677',
      ',': '#556677'
    },
    trainData: [
      ['pablo', 'painted', 'the', 'red', 'paper', 'yellow', '.', 'the', 'paper', 'is', 'now', 'orange', '.'],
      ['pablo', 'painted', 'the', 'yellow', 'paper', 'red', '.', 'the', 'paper', 'is', 'now', 'orange', '.'],
      ['pablo', 'painted', 'the', 'blue', 'paper', 'yellow', '.', 'the', 'paper', 'is', 'now', 'green', '.'],
      ['before', 'pablo', 'painted', 'the', 'paper', 'red', ',', 'it', 'was', 'blue', '.', 'the', 'paper', 'is', 'now', 'purple', '.']
    ],
    probe: ['pablo', 'painted', 'the', 'blue', 'paper', 'yellow', '.', 'the', 'paper', 'is', 'now'],
    probeTarget: 'green',
    embDim: 16,
    ctxLen: 17,
    nHeads: 2,
    ffnHid: 32,
    lr: 0.003,
    queryPlaceholder: 'pablo painted the red paper',
    stageDemo: {
      input: '"pablo painted the red paper yellow" → each word becomes a token.',
      embedding: 'Each word gets a numeric fingerprint the model can work with.',
      posenc: 'Same word in a different position gets a slightly different pattern.',
      attention: 'Heatmap shows how the model connects "painted red" to "purple".',
      ffn: 'Hidden units fire differently for each painting scenario.',
      output: 'Taller bars mean a stronger guess for the next word.'
    }
  }
};

export const DATASET_KEYS = Object.keys(DATASETS);
