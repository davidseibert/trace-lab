"""Solver-verifier RLVR on Qwen3-0.6B — GRPO with the minimax solver as the
reward, the baseline arm of INSIGHTS §5's Qwen-as-student triad.

The setup is the RLVR paradigm in miniature, with the verifier perfect,
instant, and total: sample a tic-tac-toe position, prompt Qwen with the SAME
board-state text the Tic·arena uses (tic.move_prompt), and treat the single
next token as the whole episode — +1 if the solver calls the move optimal,
0 if legal but suboptimal, -1 if illegal or off-task. One forward per batch of
positions yields the entire policy, so a GRPO "group" is G multinomial draws
from one distribution — bandit-clean: no length effects, no credit assignment.

Objective per step: -A·log pi(a) + beta·KL(pi || pi_ref), with the group-mean
baseline A = r - mean(r) (no std division — the summed/bits-shaped estimator,
per the Dr-GRPO discussion in INSIGHTS §9) and the KL computed EXACTLY over
the vocab (we hold full logits; no k3 estimation needed). The reference policy
is the base model itself — LoRA disabled is pi_ref for free.

Evaluation is the arena's report card, re-measured every eval_every steps on
the SAME seeded probe/block suites as the web UI (tic.build_probe_suite is
draw-for-draw compatible): agreement, bits-vs-optimal, illegal mass,
decisiveness, D4 equivariance, blocks%. The falsifiable prediction this run
exists to test (INSIGHTS §5.8): illegal mass falls first, agreement second,
blocks% possibly never. The step-0 eval doubles as a parity check — it should
reproduce the Qwen3-0.6B report card the arena measured (~50% agreement,
~41% illegal mass) before any training has happened.

Checkpoints are saved MERGED (LoRA folded into the base weights) into the
shared local-model dir, so `local/tic-rl-mid` / `local/tic-rl-final` appear in
the engine dropdown and every lens — and the web arena — can read them like
any other model. Events stream as dicts (train.py's generator convention), so
a future /rl SSE endpoint gets this loop unchanged; metrics also append to
<out>/tic-rl-metrics.jsonl for offline analysis.

    uv run python qwen_rl.py --steps 400            # bare metal
    docker compose run --rm --build smoke python qwen_rl.py --steps 400
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from tic import (EQUIV_SUB, TRANSFORMS, ProbePosition, analyze, board_from_key, board_from_moves,
                 board_key, build_block_suite, build_probe_suite, generate_game, is_terminal,
                 legal_moves, move_prompt, transform_moves)

OUT = Path(os.environ.get("LENS_LOCAL_DIR", "data/local"))
BITS = 1 / math.log(2)


@dataclass
class RlConfig:
    model: str = "Qwen/Qwen3-0.6B"
    steps: int = 400
    batch_positions: int = 16   # positions per step (one forward each)
    group: int = 8              # GRPO group: samples per position
    lr: float = 1e-4
    beta: float = 0.02          # KL(pi || base) coefficient, in nats
    # Entropy bonus, exact over the vocab. The baseline run (entropy 0) froze
    # at step ~150: the policy went deterministic, every group's G samples
    # came back identical, and the group-relative advantage was 0 everywhere —
    # GRPO's exploration self-terminates at the bandit level. The bonus keeps
    # pi(runner-up) alive so sampling can still surface counterevidence.
    entropy_coef: float = 0.02
    seed: int = 0
    suite_seed: int = 0         # probe/block suite seed — match the arena UI's
    eval_every: int = 25
    lora_r: int = 16
    lora_alpha: int = 32
    # Prompt serialization (tic.Encoding): 'board' is the arena's exact text;
    # 'moves' is the toy GPT's move-sequence encoding; 'both' adds the history
    # line under the board — the encoding arm of INSIGHTS §5.
    encoding: str = "board"
    # Path to a frozen reward-model table (web/scripts/export-teacher.ts).
    # Empty = the solver verifier. Set = the mapped-Goodhart arm (INSIGHTS
    # §5.9): reward is the TOY'S raw probability on Qwen's move — no longer
    # verifiable reward but RLHF, with a reward model whose every defect is
    # enumerable against the solver before training starts. Off-task tokens
    # score 0 (the reward model cannot price them).
    teacher: str = ""


# ---------------------------------------------------------------------------
# Tokenizer plumbing
# ---------------------------------------------------------------------------

def digit_token_map(tok) -> dict[int, int]:
    """Token id -> cell for every vocab entry that IS a digit 0-8 after
    trimming ('4', ' 4', '4\\n', ...) — the full-vocab version of the web's
    parseDigitProbs (which reads a top-20 truncation; the difference is
    negligible mass, but this side is exact)."""
    out: dict[int, int] = {}
    for raw, tid in tok.get_vocab().items():
        # Cheap prefilter on the raw byte-level string; ASCII digits survive
        # byte-level BPE literally, which is also exactly the set the web's
        # trim + '0' <= s <= '8' check accepts.
        if len(raw) > 4 or not any(c.isdigit() for c in raw):
            continue
        s = tok.decode([tid]).strip()
        if len(s) == 1 and "0" <= s <= "8":
            out[tid] = int(s)
    return out


def calibrate_ending(policy_probs, digit_ids, encoding: str = "board") -> str:
    """Port of llmPlayer's self-calibration: Qwen/Gemma tokenize the trailing
    space separately (want ' '), GPT-2-style BPEs fuse ' 4' (want '').
    Measured on the empty-board prompt against the BASE policy."""
    def decisiveness(ending: str) -> float:
        probs = policy_probs([move_prompt([], encoding=encoding) + ending])
        return float(probs[0, digit_ids].sum())

    bare = decisiveness("")
    if bare >= 0.05:
        return ""
    return " " if decisiveness(" ") > bare else ""


# ---------------------------------------------------------------------------
# Report card — mirror of arena.ts reportCard over the seeded suites
# ---------------------------------------------------------------------------

def transformed(p: ProbePosition, g: int) -> ProbePosition:
    return ProbePosition(
        moves=tuple(transform_moves(g, list(p.moves))),
        board=board_from_moves(transform_moves(g, list(p.moves))),
        legal=tuple(transform_moves(g, list(p.legal))),
        optimal=tuple(transform_moves(g, list(p.optimal))),
    )


def legal_policy(cell_mass: list[float], legal: tuple[int, ...]) -> tuple[list[float], float]:
    """arena.ts legalPolicy: legal-renormalized pi + digit mass on illegal
    cells; zero legal mass falls back to uniform with illegalMass 1."""
    total = sum(cell_mass)
    legal_sum = sum(cell_mass[m] for m in legal)
    if legal_sum > 0:
        pi = [0.0] * 9
        for m in legal:
            pi[m] = cell_mass[m] / legal_sum
        return pi, max(0.0, total - legal_sum)
    pi = [0.0] * 9
    for m in legal:
        pi[m] = 1 / len(legal)
    return pi, 1.0


def argmax_legal(pi: list[float], legal: tuple[int, ...]) -> int:
    arg = legal[0]
    for m in legal:
        if pi[m] > pi[arg]:
            arg = m
    return arg


def report_card(policy_cells, suite: list[ProbePosition], blocks: list[ProbePosition],
                teacher: dict[int, list[float]] | None = None) -> dict:
    """policy_cells: list of positions -> per-position ([9] digit-cell mass).
    Batches all unique boards in one pass; memoized by board key. With a
    teacher table, adds the Goodhart readouts: agreement with the TEACHER's
    argmax, and agreement with the SOLVER restricted to the suite positions
    where the teacher is wrong — the mapped cliff. The prediction is that the
    first rises while the second falls."""
    sub = suite[:EQUIV_SUB]
    wanted: dict[int, ProbePosition] = {}
    for p in [*suite, *blocks, *(transformed(p, g) for p in sub for g in range(1, 8))]:
        wanted.setdefault(board_key(p.board), p)
    order = list(wanted.values())
    masses = policy_cells(order)
    at = {board_key(p.board): m for p, m in zip(order, masses)}

    agree = bits = illegal = decisive = 0.0
    arg_at: dict[int, int] = {}
    for p in suite:
        mass = at[board_key(p.board)]
        pi, ill = legal_policy(mass, p.legal)
        illegal += ill
        decisive += sum(mass)
        arg = argmax_legal(pi, p.legal)
        arg_at[board_key(p.board)] = arg
        if arg in p.optimal:
            agree += 1
        opt_mass = sum(pi[m] for m in p.optimal)
        bits += min(10.0, -math.log2(max(opt_mass, 2**-10)))

    equiv = 0.0
    for p in sub:
        base_pi, _ = legal_policy(at[board_key(p.board)], p.legal)
        for g in range(1, 8):
            gp = transformed(p, g)
            trans_pi, _ = legal_policy(at[board_key(gp.board)], gp.legal)
            tv = sum(abs(trans_pi[TRANSFORMS[g][m]] - base_pi[m]) for m in p.legal)
            equiv += tv / 2

    blocked = 0
    for p in blocks:
        pi, _ = legal_policy(at[board_key(p.board)], p.legal)
        if argmax_legal(pi, p.legal) == p.optimal[0]:
            blocked += 1

    card = {
        "agreement": agree / len(suite),
        "bitsVsOptimal": bits / len(suite),
        "equivariance": equiv / (len(sub) * 7),
        "illegalMass": illegal / len(suite),
        "decisiveness": decisive / len(suite),
        "blocks": blocked / len(blocks) if blocks else 0.0,
    }
    if teacher is not None:
        t_agree = on_wrong = n_wrong = 0
        for p in suite:
            t_pi, _ = legal_policy(teacher[board_key(p.board)], p.legal)
            t_arg = argmax_legal(t_pi, p.legal)
            if arg_at[board_key(p.board)] == t_arg:
                t_agree += 1
            if t_arg not in p.optimal:
                n_wrong += 1
                if arg_at[board_key(p.board)] in p.optimal:
                    on_wrong += 1
        card["agreeTeacher"] = t_agree / len(suite)
        card["agreeSolverOnTeacherWrong"] = on_wrong / n_wrong if n_wrong else 1.0
        card["teacherWrongInSuite"] = n_wrong
    return card


# ---------------------------------------------------------------------------
# The GRPO loop
# ---------------------------------------------------------------------------

def sample_position(rnd: random.Random) -> list[int]:
    """A non-terminal position from a mixed-policy game (the probe suite's own
    distribution, plies 0..8) — this rng needs no web parity, only seeding."""
    policy = "optimal" if rnd.random() < 0.5 else "random"
    game = generate_game(rnd.random, policy)
    return game[: rnd.randrange(len(game))]


def train_run(cfg: RlConfig, out: Path = OUT) -> Iterator[dict]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last position = the move token for every row

    # bf16 base weights — the dtype the engine itself serves these models at
    # (so the step-0 report card matches the arena's numbers), and half the
    # VRAM of fp32 on a 12 GB laptop card that also runs the desktop (an fp32
    # run died at step ~40 to a transient CUBLAS failure). Precision lives
    # where it matters: peft keeps the LoRA adapters + Adam state in fp32
    # (autocast_adapter_dtype default), and every logit is cast .float()
    # before log_softmax, so the loss/KL/eval arithmetic is fp32 throughout.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)
    sampler = torch.Generator(device=device).manual_seed(cfg.seed)

    dmap = digit_token_map(tok)
    digit_ids = torch.tensor(sorted(dmap), device=device)
    digit_cells = torch.tensor([dmap[i] for i in sorted(dmap)], device=device)

    @torch.no_grad()
    def probs_of(prompts: list[str]) -> torch.Tensor:
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, -1]
        return F.softmax(logits.float(), dim=-1)

    ending = calibrate_ending(probs_of, digit_ids, cfg.encoding)
    prompt_of = lambda moves: move_prompt(list(moves), encoding=cfg.encoding) + ending

    def policy_cells(positions: list[ProbePosition]) -> list[list[float]]:
        masses: list[list[float]] = []
        for i in range(0, len(positions), 16):
            chunk = positions[i : i + 16]
            probs = probs_of([prompt_of(p.moves) for p in chunk])
            cells = torch.zeros(len(chunk), 9, device=device)
            cells.index_add_(1, digit_cells, probs[:, digit_ids])
            masses.extend(cells.tolist())
        return masses

    suite = build_probe_suite(cfg.suite_seed)
    blocks = build_block_suite(cfg.suite_seed)

    # The mapped-Goodhart teacher: freeze it, card it, and enumerate its
    # entire error surface against the solver BEFORE training starts.
    teacher: dict[int, list[float]] | None = None
    teacher_info: dict = {}
    if cfg.teacher:
        table = json.loads(Path(cfg.teacher).read_text(encoding="utf-8"))
        teacher = {int(k): v for k, v in table["policies"].items()}
        wrong = 0
        for k, q in teacher.items():
            b = board_from_key(k)
            legal = tuple(legal_moves(b))
            t_pi, _ = legal_policy(q, legal)
            if argmax_legal(t_pi, legal) not in analyze(b).optimal:
                wrong += 1
        teacher_info = {
            "teacher_meta": table["meta"],
            "teacher_wrong_positions": wrong,          # of len(teacher) reachable
            "teacher_card": report_card(
                lambda ps: [teacher[board_key(p.board)] for p in ps], suite, blocks),
        }
    metrics_path = out / "tic-rl-metrics.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    log = metrics_path.open("w", encoding="utf-8")

    def emit(ev: dict) -> dict:
        log.write(json.dumps(ev) + "\n")
        log.flush()
        return ev

    def save_merged(name: str, note: str) -> Path:
        # deepcopy so merge_and_unload can destroy ITS copy — training continues
        # on the live adapters. ~2.4 GB fp32 held twice, briefly; fits.
        merged = copy.deepcopy(model).merge_and_unload()
        path = out / name
        # Windows: a leftover model.safetensors from a previous run can be
        # transiently locked (AV scan, indexer) and safetensors dies on
        # os error 5 — clear the target and retry once.
        for attempt in (1, 2):
            try:
                shutil.rmtree(path, ignore_errors=True)
                merged.to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        tok.save_pretrained(path)
        (path / "lens_meta.json").write_text(json.dumps({
            "note": note, "family": "tic", "encoding": cfg.encoding,
            "prompts": [move_prompt([], encoding=cfg.encoding),
                        move_prompt([4, 0, 8], encoding=cfg.encoding)],
        }, indent=2))
        del merged
        if device == "cuda":
            torch.cuda.empty_cache()
        return path

    n_params = sum(p.numel() for p in trainable)
    yield emit({"event": "start", **asdict(cfg), "device": device,
                "trainable_params": n_params, "digit_tokens": len(dmap),
                "ending": repr(ending), **teacher_info})

    model.eval()  # no dropout anywhere; grads still flow where we ask
    yield emit({"event": "eval", "step": 0, **report_card(policy_cells, suite, blocks, teacher)})

    rnd = random.Random(cfg.seed)
    for step in range(1, cfg.steps + 1):
        positions = [sample_position(rnd) for _ in range(cfg.batch_positions)]
        boards = [board_from_moves(m) for m in positions]
        enc = tok([prompt_of(m) for m in positions], return_tensors="pt", padding=True).to(device)

        logits = model(**enc).logits[:, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        with torch.no_grad(), model.disable_adapter():
            ref_logp = F.log_softmax(model(**enc).logits[:, -1].float(), dim=-1)

        actions = torch.multinomial(probs.detach(), cfg.group, replacement=True, generator=sampler)

        rewards = torch.empty_like(actions, dtype=torch.float32)
        off_task = illegal = 0
        for i, (b, moves) in enumerate(zip(boards, positions)):
            optimal = set(analyze(b).optimal)
            q = teacher[board_key(b)] if teacher is not None else None
            for j in range(cfg.group):
                cell = dmap.get(int(actions[i, j]))
                if cell is None:
                    off_task += 1
                elif b[cell] != 0:
                    illegal += 1
                if q is not None:
                    # RLHF-faithful: the reward model prices every cell it can
                    # see, occupied ones included — its defects are the point.
                    rewards[i, j] = 0.0 if cell is None else q[cell]
                elif cell is None or b[cell] != 0:
                    rewards[i, j] = -1.0
                else:
                    rewards[i, j] = 1.0 if cell in optimal else 0.0

        advantage = rewards - rewards.mean(dim=1, keepdim=True)
        pg = -(advantage * logp.gather(1, actions)).mean()
        kl = (probs * (logp - ref_logp)).sum(-1).mean()  # exact, in nats
        entropy = -(probs * logp).sum(-1).mean()         # exact, in nats
        loss = pg + cfg.beta * kl - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        n = cfg.batch_positions * cfg.group
        yield emit({"event": "step", "step": step, "loss": round(float(loss.detach()), 5),
                    "reward": round(float(rewards.mean()), 4),
                    "off_task": round(off_task / n, 4), "illegal": round(illegal / n, 4),
                    "kl_bits": round(float(kl.detach()) * BITS, 5),
                    "entropy_bits": round(float(entropy.detach()) * BITS, 5)})

        if step % cfg.eval_every == 0 or step == cfg.steps:
            yield emit({"event": "eval", "step": step,
                        **report_card(policy_cells, suite, blocks, teacher)})
        if step == cfg.steps // 2:
            # A checkpoint is a convenience; its I/O failure must not cost the run.
            try:
                path = save_merged("tic-rl-mid", f"{'teacher' if teacher else 'solver'}-RLVR, step {step}/{cfg.steps}")
                yield emit({"event": "checkpoint", "name": "tic-rl-mid", "path": str(path)})
            except Exception as e:
                yield emit({"event": "checkpoint_error", "name": "tic-rl-mid", "detail": str(e)})

    path = save_merged("tic-rl-final", f"{'teacher' if teacher else 'solver'}-RLVR, {cfg.steps} steps, beta={cfg.beta}")
    yield emit({"event": "checkpoint", "name": "tic-rl-final", "path": str(path)})
    yield emit({"event": "done", "steps": cfg.steps})
    log.close()


EVAL_COLS = ["agreement", "bitsVsOptimal", "equivariance", "illegalMass", "decisiveness", "blocks"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f, default in ((f.name, f.default) for f in RlConfig.__dataclass_fields__.values()):
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    cfg = RlConfig(**vars(ap.parse_args()))

    for ev in train_run(cfg):
        kind = ev["event"]
        if kind == "start":
            print(f"{ev['model']} on {ev['device']}: {ev['trainable_params'] / 1e6:.2f}M LoRA params, "
                  f"{ev['digit_tokens']} digit tokens, ending={ev['ending']}")
        elif kind == "step" and ev["step"] % 10 == 0:
            print(f"step {ev['step']:4d}  loss {ev['loss']:+.4f}  reward {ev['reward']:+.3f}  "
                  f"off-task {ev['off_task']:.1%}  illegal {ev['illegal']:.1%}  KL {ev['kl_bits']:.3f}b  "
                  f"H {ev['entropy_bits']:.3f}b")
        elif kind == "eval":
            cols = EVAL_COLS + (["agreeTeacher", "agreeSolverOnTeacherWrong"] if "agreeTeacher" in ev else [])
            print(f"eval @ {ev['step']:4d}  " + "  ".join(f"{c} {ev[c]:.3f}" for c in cols))
        elif kind == "checkpoint":
            print(f"saved {ev['path']}")
        elif kind == "done":
            print(f"done: {ev['steps']} steps")


if __name__ == "__main__":
    main()
