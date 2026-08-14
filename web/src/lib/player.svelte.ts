/**
 * Playback engine for a trace. The trace is immutable; this just holds an
 * index into it plus play/pause/speed, so the UI is a pure function of
 * `current`. This is what makes the app feel like a debugger: stepping,
 * scrubbing backward, and slow-motion are all index manipulation.
 */

import { untrack } from 'svelte';

/**
 * Generic over the step element type `S`: the player only ever indexes into
 * `steps[]`, so it neither knows nor cares what a step contains. The MDL lens
 * loads `Step<GrammarModel, DigramMove>[]`; the transformer lens loads
 * `LlmStep[]`. Same playback, two algorithms.
 */
export class Player<S> {
  steps = $state<S[]>([]);
  index = $state(0);
  playing = $state(false);
  /** Steps per second when playing. */
  speed = $state(1.5);

  #timer: ReturnType<typeof setTimeout> | null = null;

  current = $derived(this.steps[this.index]);
  count = $derived(this.steps.length);
  atEnd = $derived(this.index >= this.steps.length - 1);
  atStart = $derived(this.index <= 0);

  load(steps: S[]) {
    this.pause();
    this.steps = steps;
    this.index = 0;
  }

  /**
   * Swap in a recomputed trace but KEEP the current position (clamped) —
   * unlike load(), which rewinds to 0. For lenses whose steps are re-derived
   * from upstream controls while the playback cursor should hold still.
   * Untracked so a caller inside an $effect depends only on the new steps it
   * passes in, not on the playback position.
   */
  reload(steps: S[]) {
    untrack(() => {
      this.steps = steps;
      this.seek(this.index);
    });
  }

  /**
   * Push one streamed step. If the cursor was sitting at the end (i.e. the
   * user was following the live stream, not scrubbing back), advance it to
   * the new last step; pass `follow = false` to never advance.
   */
  append(step: S, follow = true) {
    const wasFollowing = this.atEnd;
    this.steps.push(step);
    if (follow && wasFollowing) this.index = this.steps.length - 1;
  }

  seek(i: number) {
    this.index = Math.max(0, Math.min(this.steps.length - 1, i));
  }

  stepForward() {
    if (!this.atEnd) this.index++;
    else this.pause();
  }

  stepBack() {
    if (!this.atStart) this.index--;
  }

  reset() {
    this.pause();
    this.index = 0;
  }

  play() {
    if (this.playing || this.steps.length === 0) return;
    if (this.atEnd) this.index = 0;
    this.playing = true;
    this.#tick();
  }

  pause() {
    this.playing = false;
    if (this.#timer) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  setSpeed(s: number) {
    this.speed = s;
    if (this.playing) {
      this.pause();
      this.play();
    }
  }

  #tick = () => {
    if (!this.playing) return;
    if (this.atEnd) {
      this.pause();
      return;
    }
    this.index++;
    this.#timer = setTimeout(this.#tick, 1000 / this.speed);
  };
}
