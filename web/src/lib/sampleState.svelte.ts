/**
 * Shared preamble for the concept lenses (grammar / morphology / morfessor /
 * graph). Each lens owns a SAMPLES record and a default; this class holds the
 * URL-backed settings they all share — which sample, the (possibly edited)
 * input text, the code mode, and the overhead toggle — and wires the $effect
 * that mirrors them into the hash query. Default values map to `null` so they
 * stay out of the URL: a bare `#/lens` link always means factory settings.
 *
 * Must be constructed during component init (the constructor creates an
 * $effect). Settings live in the URL: refresh keeps them, and the URL is
 * shareable.
 */
import { router } from './router.svelte';

export type CodeMode = 'uniform' | 'shannon';

export interface LensSettingsOptions {
  samples: Record<string, string>;
  defaultSample: string;
  /** The code mode a bare URL means for this lens (default 'uniform'). */
  defaultCodeMode?: CodeMode;
  /**
   * Whether this lens prices the model-of-the-model (adds the `oh` query
   * key). Default true; the morfessor lens has no such toggle.
   */
  overhead?: boolean;
}

export class LensSettings {
  readonly samples: Record<string, string>;

  sampleKey: string = $state('');
  text: string = $state('');
  codeMode: CodeMode = $state('uniform');
  includeOverhead: boolean = $state(true);

  constructor(opts: LensSettingsOptions) {
    const { samples, defaultSample, defaultCodeMode = 'uniform', overhead = true } = opts;
    this.samples = samples;

    const urlSample = router.get('sample');
    const initialSample = urlSample && samples[urlSample] ? urlSample : defaultSample;
    this.sampleKey = initialSample;
    this.text = router.get('text') ?? samples[initialSample];
    const altMode: CodeMode = defaultCodeMode === 'uniform' ? 'shannon' : 'uniform';
    this.codeMode = router.get('code') === altMode ? altMode : defaultCodeMode;
    this.includeOverhead = overhead ? (router.bool('oh') ?? true) : true;

    $effect(() => {
      router.setQuery({
        sample: this.sampleKey === defaultSample ? null : this.sampleKey,
        text: this.text === samples[this.sampleKey] ? null : this.text,
        code: this.codeMode === defaultCodeMode ? null : this.codeMode,
        ...(overhead ? { oh: this.includeOverhead ? null : false } : {})
      });
    });
  }

  /** Switch sample and reset the editable text to that sample's data. */
  pick = (k: string) => {
    this.sampleKey = k;
    this.text = this.samples[k];
  };
}
