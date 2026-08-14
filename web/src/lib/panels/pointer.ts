/**
 * Tiny pointer-tracking helpers shared by the resize gutters and the panel
 * drag controller. Window-level listeners so the drag survives leaving the
 * element; Escape and pointercancel both cancel.
 */

export interface TrackOpts {
  onMove?: (dx: number, dy: number, ev: PointerEvent) => void;
  /** Called exactly once when the interaction ends. */
  onEnd?: (cancelled: boolean) => void;
}

export function trackPointer(e: PointerEvent, opts: TrackOpts): void {
  const x0 = e.clientX;
  const y0 = e.clientY;
  let done = false;

  const move = (ev: PointerEvent) =>
    opts.onMove?.(ev.clientX - x0, ev.clientY - y0, ev);
  const up = () => finish(false);
  const cancel = () => finish(true);
  const key = (ev: KeyboardEvent) => {
    if (ev.key === 'Escape') {
      ev.stopPropagation();
      finish(true);
    }
  };

  function finish(cancelled: boolean) {
    if (done) return;
    done = true;
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', cancel);
    window.removeEventListener('keydown', key, true);
    opts.onEnd?.(cancelled);
  }

  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', cancel);
  // Capture phase so a mid-drag Escape doesn't also unfocus/close overlays.
  window.addEventListener('keydown', key, true);
}

/**
 * Resize a pair of flex siblings by dragging the gutter between them.
 * Snapshots pixel sizes once, converts the drag delta into new weights with a
 * pixel floor for both members, applies live, commits once on release.
 */
export function resizePair(
  e: PointerEvent,
  opts: {
    axis: 'x' | 'y';
    aPx: number;
    bPx: number;
    aWeight: number;
    bWeight: number;
    minPx: number;
    apply: (aW: number, bW: number) => void;
    commit: () => void;
  }
): void {
  const total = opts.aPx + opts.bPx;
  const wsum = opts.aWeight + opts.bWeight;
  if (total <= 0 || wsum <= 0) return;
  // Both members must keep minPx; if they can't, the gutter is inert.
  if (total < opts.minPx * 2) return;

  trackPointer(e, {
    onMove: (dx, dy) => {
      const d = opts.axis === 'x' ? dx : dy;
      const a = Math.min(total - opts.minPx, Math.max(opts.minPx, opts.aPx + d));
      const aW = (wsum * a) / total;
      opts.apply(aW, wsum - aW);
    },
    onEnd: (cancelled) => {
      if (cancelled) opts.apply(opts.aWeight, opts.bWeight);
      else opts.commit();
    }
  });
}
