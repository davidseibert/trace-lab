"""One-shot CLI sanity check: print the bits ladder for a prompt.

    make smoke                          # gpt2, default prompt
    make smoke MODEL=gpt2-large PROMPT="1 + 100 ="

The ladder is the whole app in one column of numbers: -log2 p of the model's
final prediction at every rung, classic lens vs J-lens, shrinking with depth.
"""
from __future__ import annotations

import os
import sys

from lens import lens_report, load_model

model_name = os.environ.get("LENS_MODEL", "gpt2")
prompt = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROMPT", "The Eiffel Tower is in the city of")

model, tok, device = load_model(model_name)
r = lens_report(model, tok, prompt, top_k=1, jlens=True)

print(f"\n{model_name} on {device} — {prompt!r}")
print(f"prediction: {r.pred['token']!r}  p={r.pred['p']:.3f}  ({r.pred['bits']:.2f} bits; uniform {r.uniform:.1f})\n")
print(f"{'rung':>10}  {'lens':>7}  {'J-lens':>7}  top-1 (classic)")
# Rungs printed top-down (final first), so reading downward the bits GROW back
# toward the embed rung — the compression story in reverse.
for i in range(len(r.layers) - 1, -1, -1):
    top = r.grid[i][-1][0]  # rung i, last column, top-1
    # jbits can be None: lens.py's _ladder falls back to the classic-only path
    # when the J-lens degrades. The classic ladder still prints.
    jb = f"{r.jbits[i]:6.2f}b" if r.jbits else "   —   "
    print(f"{r.layers[i]:>10}  {r.bits[i]:6.2f}b  {jb}  {top['t']!r} ({top['p']:.1%})")
