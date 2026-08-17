"""Report-card any model under any prompt encoding — the measurement half of
the encoding arm (INSIGHTS §5): does serializing the game as a move sequence
(the toy GPT's encoding, where the opponent's threat is the newest token)
shift the card relative to the arena's board-state prompt, on a FROZEN model?

One line of JSON per (model, encoding) pair, same columns as the arena's
report card, same seeded suites (tic.build_probe_suite is draw-for-draw
compatible with the web).

    uv run python tic_card.py                                   # base Qwen
    uv run python tic_card.py --model data/local/tic-rl-final   # a checkpoint
    uv run python tic_card.py --encodings board moves both
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from qwen_rl import calibrate_ending, digit_token_map, report_card
from tic import build_block_suite, build_probe_suite, move_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--encodings", nargs="+", default=["board", "moves", "both"])
    ap.add_argument("--suite-seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    dmap = digit_token_map(tok)
    digit_ids = torch.tensor(sorted(dmap), device=device)
    digit_cells = torch.tensor([dmap[i] for i in sorted(dmap)], device=device)

    @torch.no_grad()
    def probs_of(prompts: list[str]) -> torch.Tensor:
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        return F.softmax(model(**enc).logits[:, -1].float(), dim=-1)

    suite = build_probe_suite(args.suite_seed)
    blocks = build_block_suite(args.suite_seed)

    for encoding in args.encodings:
        ending = calibrate_ending(probs_of, digit_ids, encoding)

        def policy_cells(positions, _e=encoding, _end=ending):
            masses: list[list[float]] = []
            for i in range(0, len(positions), 16):
                chunk = positions[i : i + 16]
                probs = probs_of([move_prompt(list(p.moves), encoding=_e) + _end for p in chunk])
                cells = torch.zeros(len(chunk), 9, device=device)
                cells.index_add_(1, digit_cells, probs[:, digit_ids])
                masses.extend(cells.tolist())
            return masses

        card = report_card(policy_cells, suite, blocks)
        print(json.dumps({"model": args.model, "encoding": encoding, "ending": repr(ending),
                          **{k: round(v, 4) for k, v in card.items()}}))


if __name__ == "__main__":
    main()
