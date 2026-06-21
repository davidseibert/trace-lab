/**
 * Playback engine for a trace. The trace is immutable; this just holds an
 * index into it plus play/pause/speed, so the UI is a pure function of
 * `current`. This is what makes the app feel like a debugger: stepping,
 * scrubbing backward, and slow-motion are all index manipulation.
 */

import type { Step } from './mdl/types';

export class Player<Model, Move> {
  steps = $state<Step<Model, Move>[]>([]);
  index = $state(0);
  playing = $state(false);
  /** Steps per second when playing. */
  speed = $state(1.5);

  #timer: ReturnType<typeof setInterval> | null = null;

  current = $derived(this.steps[this.index]);
  count = $derived(this.steps.length);
  atEnd = $derived(this.index >= this.steps.length - 1);
  atStart = $derived(this.index <= 0);

  load(steps: Step<Model, Move>[]) {
    this.pause();
    this.steps = steps;
    this.index = 0;
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
