import { describe, expect, test } from 'bun:test';
import {
  binaryStream,
  crossEntropyBits,
  effectiveChoices,
  entropyBits,
  informationBits,
  klDivergenceBits,
  normalize
} from './derive';

describe('math foundations derivations', () => {
  test('probability landmarks map to whole bits', () => {
    expect(informationBits(1)).toBe(0);
    expect(informationBits(1 / 2)).toBe(1);
    expect(informationBits(1 / 8)).toBe(3);
    expect(informationBits(0)).toBe(Infinity);
  });

  test('entropy and effective choices agree for a fair four-way choice', () => {
    const h = entropyBits([0.25, 0.25, 0.25, 0.25]);
    expect(h).toBeCloseTo(2);
    expect(effectiveChoices(h)).toBeCloseTo(4);
  });

  test('cross entropy decomposes into entropy plus KL', () => {
    const q = [0.75, 0.25];
    const p = [0.6, 0.4];
    expect(crossEntropyBits(q, p)).toBeCloseTo(entropyBits(q) + klDivergenceBits(q, p));
    expect(klDivergenceBits(q, q)).toBeCloseTo(0);
  });

  test('normalization and binary streams produce distributions', () => {
    expect(normalize([2, 3, 5])).toEqual([0.2, 0.3, 0.5]);
    const stream = binaryStream('AB', 0.7);
    expect(stream[0].dist.reduce((s, x) => s + x.p, 0)).toBeCloseTo(1);
    expect(stream.map((x) => x.chosenIndex)).toEqual([0, 1]);
  });
});
