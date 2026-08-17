import type { MathLesson } from './types';

export const MATH_LESSONS: MathLesson[] = [
  {
    id: 'binary-choices',
    chapter: 'the house currency',
    title: 'Choices and binary numbers',
    shortTitle: 'Binary choices',
    question: 'Why can a number of choices be measured in bits?',
    summary: 'A bit answers one yes/no question. A string of b bits names 2^b possibilities, so naming one of N equally likely possibilities ideally takes log2(N) bits.',
    prerequisites: [],
    terms: [
      { symbol: 'b', meaning: 'number of binary digits', unit: 'bits' },
      { symbol: 'N', meaning: 'number of possible choices', unit: 'choices' }
    ],
    blocks: [
      { kind: 'text', title: 'Start with counting', text: 'One bit has two values: 0 or 1. Two bits have four strings. Three bits have eight. Each added bit doubles the number of distinguishable cases.' },
      { kind: 'equation', lines: ['b bits  ->  2^b distinct strings', '2^b = N', 'b = log2(N)'] },
      { kind: 'text', title: 'What about fractional bits?', text: 'If N is not a power of two, log2(N) is an ideal average length. A single standalone codeword still has an integer number of digits; long messages and arithmetic coding let many outcomes share the rounding cost.' },
      { kind: 'callout', title: 'Reading the app', text: 'The uniform-code label log2(V) means: choose one symbol from a vocabulary of V equally likely symbols.' }
    ],
    lab: 'probability-bits',
    seenIn: [
      { lens: 'grammar', label: 'Grammar', note: 'uniform log2(V) code toggle' },
      { lens: 'coder', label: 'Coder', note: 'uniform source and literal binary codeword' }
    ]
  },
  {
    id: 'logarithms',
    chapter: 'the house currency',
    title: 'Logarithms without mystery',
    shortTitle: 'Logarithms',
    question: 'What is a logarithm doing in all these formulas?',
    summary: 'A logarithm asks for an exponent. Its crucial gift is turning multiplication into addition, which turns a product of probabilities into an accumulating bit bill.',
    prerequisites: ['binary-choices'],
    terms: [
      { symbol: 'log2(x)', meaning: 'the exponent that makes 2^b equal x', unit: 'bits when used as a code length' },
      { symbol: 'ln(x)', meaning: 'logarithm base e', unit: 'nats in information formulas' }
    ],
    blocks: [
      { kind: 'equation', title: 'Inverse of exponentiation', lines: ['2^3 = 8  <=>  log2(8) = 3', '2^-3 = 1/8  <=>  -log2(1/8) = 3'] },
      { kind: 'equation', title: 'The rule the app depends on', lines: ['log2(a * b) = log2(a) + log2(b)', '-log2(product_i p_i) = sum_i -log2(p_i)'] },
      { kind: 'text', title: 'Changing units', text: 'The log base chooses the unit, not the underlying uncertainty. Base 2 gives bits. Natural logs give nats. Divide nats by ln(2) to get bits.' },
      { kind: 'callout', title: 'Numerical reason', text: 'Products of hundreds of small probabilities underflow toward zero. Adding their log probabilities is both interpretable and numerically safer.', tone: 'good' }
    ],
    lab: 'probability-bits',
    seenIn: [
      { lens: 'reason', label: 'Reason·trace', note: 'token probabilities become additive code lengths' },
      { lens: 'train', label: 'Train·real', note: 'natural-log training loss versus bit readouts' }
    ]
  },
  {
    id: 'probability',
    chapter: 'the house currency',
    title: 'Probability and conditional probability',
    shortTitle: 'Probability',
    question: 'What exactly is the p inside -log2(p)?',
    summary: 'A distribution assigns nonnegative mass summing to one. In a sequence model, p is conditional: it is the probability of this next token given the tokens already seen.',
    prerequisites: ['logarithms'],
    terms: [
      { symbol: 'p(x)', meaning: 'probability assigned to outcome x', unit: 'number from 0 to 1' },
      { symbol: 'p(x|c)', meaning: 'probability of x given context c', unit: 'number from 0 to 1' }
    ],
    blocks: [
      { kind: 'list', title: 'A valid discrete distribution', items: ['Every p(x) is at least 0.', 'The probabilities over all possible outcomes sum to 1.', 'The probabilities describe uncertainty before the outcome is known.'] },
      { kind: 'equation', title: 'Sequence chain rule', lines: ['p(x1, x2, x3) = p(x1) p(x2|x1) p(x3|x1,x2)', 'p(sequence) = product_i p(x_i | x_<i)'] },
      { kind: 'text', title: 'The model is the codebook', text: 'At every token position, an autoregressive model provides the next conditional distribution. An encoder and decoder that share those distributions can run the same arithmetic-coding steps.' },
      { kind: 'callout', title: 'Do not confuse probability and sampling', text: 'Temperature can change which token is drawn. Reason·trace deliberately prices the drawn token under the model\'s original distribution, so path selection and price remain separate.', tone: 'warn' }
    ],
    lab: 'probability-bits',
    seenIn: [
      { lens: 'llm', label: 'Mini·GPT', note: 'next-token probability bars' },
      { lens: 'reason', label: 'Reason·trace', note: 'conditional probability at emission' }
    ]
  },
  {
    id: 'surprisal',
    chapter: 'the house currency',
    title: 'Surprisal: information in one event',
    shortTitle: 'Surprisal',
    question: 'Why does a less likely event cost more bits?',
    summary: 'Surprisal is the ideal information content of one realized event: I(x) = -log2 p(x). Certain outcomes say nothing new; rare outcomes distinguish many alternatives.',
    prerequisites: ['probability', 'logarithms'],
    terms: [
      { symbol: 'I(x)', meaning: 'self-information or surprisal of observed x', unit: 'bits' },
      { symbol: 'p(x)', meaning: 'probability assigned before x was observed' }
    ],
    blocks: [
      { kind: 'equation', title: 'Counting derivation', lines: ['event occupies fraction p of the possibilities', 'effective alternatives = 1/p', 'I(x) = log2(1/p(x)) = -log2 p(x)'] },
      { kind: 'equation', title: 'Landmarks', lines: ['p = 1     ->  0 bits', 'p = 1/2   ->  1 bit', 'p = 1/8   ->  3 bits', 'p = 1/100 ->  6.64 bits'] },
      { kind: 'text', title: 'Why surprisals add', text: 'Independent or conditionally chained probabilities multiply. Because logarithms turn multiplication into addition, the cost of a sequence is the sum of its token surprisals.' },
      { kind: 'callout', title: 'A spike is not an error', text: 'High surprisal only says the chosen token had low assigned probability. Several good continuations, creative freedom, or an unlucky sample can all create a spike.', tone: 'warn' }
    ],
    lab: 'probability-bits',
    seenIn: [
      { lens: 'reason', label: 'Reason·trace', note: 'yellow per-token shading' },
      { lens: 'logit', label: 'Logit·real', note: 'per-layer target-token code length' },
      { lens: 'arena', label: 'Tic·arena', note: 'bits paid for the played move' }
    ]
  },
  {
    id: 'entropy',
    chapter: 'the house currency',
    title: 'Entropy: average surprisal',
    shortTitle: 'Entropy',
    question: 'Why is entropy measured in bits?',
    summary: 'Entropy is the probability-weighted average of event surprisals. Because each surprisal uses log base 2, the average is measured in bits per draw.',
    prerequisites: ['surprisal'],
    terms: [
      { symbol: 'H(p)', meaning: 'entropy of distribution p', unit: 'bits/draw' },
      { symbol: '2^H', meaning: 'effective number of equally likely choices', unit: 'choices' }
    ],
    blocks: [
      { kind: 'equation', title: 'Expected surprisal', lines: ['H(p) = E_p[I(X)]', 'H(p) = sum_x p(x) [-log2 p(x)]', 'H(p) = -sum_x p(x) log2 p(x)'] },
      { kind: 'text', title: 'Event versus distribution', text: 'Surprisal belongs to the outcome that happened. Entropy belongs to the whole distribution before the draw. One rare outcome can have high surprisal even when the distribution has low entropy.' },
      { kind: 'equation', title: 'Bounds', lines: ['0 <= H(p) <= log2(V)', 'H = 0 when one outcome is certain', 'H = log2(V) for V equal probabilities'] },
      { kind: 'callout', title: 'Effective choices', text: '2^H translates entropy back into a count. An attention row with H = 2 bits mixes about as broadly as four equally weighted positions.', tone: 'good' }
    ],
    lab: 'entropy',
    seenIn: [
      { lens: 'hopfield', label: 'Hopfield·retrieve', note: 'entropy and 2^H effective patterns' },
      { lens: 'hopfieldreal', label: 'Hopfield·heads', note: 'head retrieval regimes' }
    ]
  },
  {
    id: 'codes',
    chapter: 'the house currency',
    title: 'Codeword lengths and arithmetic coding',
    shortTitle: 'Codes',
    question: 'How can fractional ideal lengths turn into actual binary digits?',
    summary: 'A decodable code spends binary-tree capacity. Arithmetic coding assigns one interval to a whole sequence, so symbols share the integer rounding and the total approaches the sum of -log2 probabilities.',
    prerequisites: ['entropy'],
    terms: [
      { symbol: 'l_x', meaning: 'binary codeword length for outcome x', unit: 'bits' },
      { symbol: 'width', meaning: 'remaining arithmetic-coding interval width' }
    ],
    blocks: [
      { kind: 'equation', title: 'Prefix-code capacity', lines: ['sum_x 2^(-l_x) <= 1      (Kraft inequality)', 'ideal l_x = -log2 p(x)', 'sum_x 2^(-l_x) = sum_x p(x) = 1'] },
      { kind: 'equation', title: 'Arithmetic-coding invariant', lines: ['new width = old width * p(chosen)', 'final width = product_i p(x_i | x_<i)', '-log2(final width) = sum_i -log2 p(x_i | x_<i)'] },
      { kind: 'text', title: 'What must be shared', text: 'The decoder needs the same ordered distributions and the message length or another stopping rule. Those are side information; a probability model is only a usable code when encoder and decoder both have it.' },
      { kind: 'callout', title: 'Ideal versus literal bits', text: 'The fractional sum is an ideal length. The demo chooses a finite binary fraction inside the final interval. Its framing is simplified and the symbol count is given to the decoder.', tone: 'warn' }
    ],
    lab: 'arithmetic',
    seenIn: [
      { lens: 'coder', label: 'Coder', note: 'full encode/decode workbench' },
      { lens: 'grammar', label: 'Grammar', note: 'ideal Shannon lengths and model overhead' }
    ]
  },
  {
    id: 'cross-entropy',
    chapter: 'the house currency',
    title: 'Cross-entropy, KL, and training loss',
    shortTitle: 'Cross-entropy',
    question: 'Why is cross-entropy loss also a compression cost?',
    summary: 'Cross-entropy is the expected number of bits paid when reality follows q but the code uses model p. It equals unavoidable entropy plus the model-mismatch penalty KL(q||p).',
    prerequisites: ['codes'],
    terms: [
      { symbol: 'q', meaning: 'target or data distribution' },
      { symbol: 'p', meaning: 'model distribution' },
      { symbol: 'H(q,p)', meaning: 'cross-entropy of q under p', unit: 'bits/draw' },
      { symbol: 'KL(q||p)', meaning: 'extra expected cost from using p instead of q', unit: 'bits/draw' }
    ],
    blocks: [
      { kind: 'equation', title: 'Expected model code length', lines: ['H(q,p) = E_q[-log2 p(X)]', 'H(q,p) = -sum_x q(x) log2 p(x)'] },
      { kind: 'equation', title: 'The decomposition', lines: ['H(q,p) = H(q) + KL(q||p)', 'KL(q||p) >= 0', 'minimum occurs at p = q'] },
      { kind: 'equation', title: 'One observed class', lines: ['q(y) = 1 and q(other) = 0', 'cross-entropy = -log2 p(y)', 'gradient with respect to logit j = p_j - 1[j=y]'] },
      { kind: 'text', title: 'Mean versus total', text: 'Mean cross-entropy is bits/token when base-2 logs are used. Multiply by the number of priced tokens to obtain a sequence or dataset description length. Training libraries commonly report natural-log loss in nats/token.' },
      { kind: 'callout', title: 'Why training can work', text: 'The gradient has a simple direction: predicted probability minus target probability. It raises underpredicted target logits and lowers overpredicted alternatives. Backpropagation carries that correction through every earlier operation.', tone: 'good' }
    ],
    lab: 'cross-entropy',
    seenIn: [
      { lens: 'llm', label: 'Mini·GPT', note: 'training loss and token probabilities' },
      { lens: 'train', label: 'Train·real', note: 'natural-log loss over answer digits' },
      { lens: 'tictac', label: 'Tic·tac', note: 'one-hot, solver-soft, and distillation targets' }
    ]
  }
];

export const mathLessonById = (id: string | null): MathLesson | undefined =>
  id ? MATH_LESSONS.find((lesson) => lesson.id === id) : undefined;
