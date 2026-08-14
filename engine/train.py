"""Train a tiny GPT-2 on two-digit addition, into the shared model volume.

The tinygrad transformer example's dataset, HF-native: every "a+b=" for
a, b in 0..99, spelled one character per token ("17+25=042" → 9 tokens over the
vocab "0123456789+="). Loss lands only on the three answer digits, so the model
must route both addends through attention — including the carry. 8,000 sums
train; 2,000 are held out, so the reported accuracy is generalization, not
recall (the model has never seen those sums).

Three checkpoints land in data/local/ (inside the shared hf-cache volume under
Docker) and appear in the engine's model dropdown as local/add-step0 /
add-mid / add-final — untrained, first ≥50% accuracy, converged — so the J-lens
gets a training-time axis over a *real* transformers model. Each checkpoint
also gets a lens_meta.json (note + sample prompts + family), which /health
surfaces so the UI pickers can label them and swap in prompts that fit.

Answers are fixed-width, most significant digit first: prompt "17+25=" and the
model should answer '0', '4', '2'.

The run itself is a generator (`train_run`) so the engine's /train endpoint can
stream it as SSE — the Train·real lens watches the same events this script
prints. Stopping a stream mid-run keeps the checkpoints saved so far.

    make train                 # Docker, GPU (EPOCHS=... to override)
    uv run python train.py     # bare metal — NOTE: saves to engine/data/local,
                               # a different filesystem than the Docker volume;
                               # train where you serve.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator

import torch
from tokenizers import Regex, Tokenizer
from tokenizers import models as tok_models
from tokenizers import pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

SEED = 0
OUT = Path(os.environ.get("LENS_LOCAL_DIR", "data/local"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = 256
LR = 3e-4

# '?' is the unk target so a stray character in a UI prompt (a space, a letter)
# degrades to a visible '?' token instead of a tokenizer exception.
DIGITS = "0123456789+="
VOCAB = DIGITS + "?"
PROMPT_LEN = 6  # "a1 a0 + b1 b0 =" — loss covers only the 3 answer digits after

# Sample prompts that actually fit this family's vocab and 16-position budget —
# saved into each checkpoint's lens_meta.json and surfaced by /health so UI
# pickers can swap them in (the default Eiffel prompt overflows n_positions).
SAMPLE_PROMPTS = ["17+25=", "99+99=", "07+08="]


def build_tokenizer() -> PreTrainedTokenizerFast:
    tk = Tokenizer(tok_models.WordLevel({c: i for i, c in enumerate(VOCAB)}, unk_token="?"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tk, model_max_length=16)


def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [f"{a:02d}+{b:02d}={a + b:03d}" for a in range(100) for b in range(100)]
    random.Random(SEED).shuffle(seqs)
    ids = torch.tensor([[VOCAB.index(c) for c in s] for s in seqs], dtype=torch.long)
    return ids[:8000], ids[8000:]


@torch.no_grad()
def exact_match(model, test: torch.Tensor) -> float:
    """Greedy-decode the 3 answer digits for every held-out sum at once."""
    model.eval()
    x = test[:, :PROMPT_LEN].clone()
    for _ in range(3):
        logits = model(input_ids=x).logits[:, -1]
        x = torch.cat([x, logits.argmax(-1, keepdim=True)], dim=1)
    ok = (x[:, PROMPT_LEN:] == test[:, PROMPT_LEN:]).all(dim=1)
    model.train()
    return float(ok.float().mean())


def save(model, tokenizer, name: str, note: str, out: Path = OUT) -> None:
    path = out / name
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # Persist the note NEXT TO the weights — /health reads it back so the
    # model pickers can say what each checkpoint is instead of a bare name.
    (path / "lens_meta.json").write_text(
        json.dumps({"note": note, "family": "add", "prompts": SAMPLE_PROMPTS}, indent=2)
    )


def train_run(epochs: int = EPOCHS, out: Path = OUT) -> Iterator[dict]:
    """The training loop as an event stream.

    Yields dicts: start / batch / epoch / checkpoint / done — consumed by the
    engine's /train SSE endpoint (the Train·real lens) and by the CLI wrapper
    below. Closing the generator mid-run (client disconnect) simply stops
    training; checkpoints already saved stay on disk.
    """
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer()
    train, test = make_data()
    train, test = train.to(device), test.to(device)

    cfg = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=16,
        n_embd=128,
        n_layer=6,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    yield {
        "event": "start",
        "params_m": round(n_params / 1e6, 2),
        "device": device,
        "epochs": epochs,
        "n_train": len(train),
        "n_test": len(test),
    }

    note0 = "untrained — the J-lens ladder should be garbage"
    save(model, tokenizer, "add-step0", note0, out)
    yield {"event": "checkpoint", "name": "add-step0", "note": note0}

    labels = train.clone()
    labels[:, :PROMPT_LEN] = -100  # loss only on the answer digits
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    n_batches = (len(train) + BATCH - 1) // BATCH
    mid_saved = False
    acc = 0.0
    epoch = 0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train), device=device)
        total = 0.0
        for b, i in enumerate(range(0, len(train), BATCH)):
            idx = perm[i : i + BATCH]
            out_ = model(input_ids=train[idx], labels=labels[idx])
            out_.loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(out_.loss) * len(idx)
            if b % 8 == 0:
                yield {
                    "event": "batch",
                    "epoch": epoch,
                    "frac": round((b + 1) / n_batches, 3),
                    "loss": round(float(out_.loss), 4),
                }
        acc = exact_match(model, test)
        yield {
            "event": "epoch",
            "epoch": epoch,
            "loss": round(total / len(train), 4),
            "acc": round(acc, 4),
        }
        if not mid_saved and acc >= 0.5:
            note = f"first ≥50% checkpoint (epoch {epoch}, {acc:.1%})"
            save(model, tokenizer, "add-mid", note, out)
            yield {"event": "checkpoint", "name": "add-mid", "note": note}
            mid_saved = True
        if acc >= 0.999:
            break

    if not mid_saved:
        note = "never crossed 50% — saved final state as mid"
        save(model, tokenizer, "add-mid", note, out)
        yield {"event": "checkpoint", "name": "add-mid", "note": note}
    note = f"final ({acc:.1%} held-out exact-match)"
    save(model, tokenizer, "add-final", note, out)
    yield {"event": "checkpoint", "name": "add-final", "note": note}
    yield {"event": "done", "acc": round(acc, 4), "epochs_run": epoch}


def main() -> None:
    for ev in train_run():
        if ev["event"] == "start":
            print(f"{ev['params_m']:.2f}M params on {ev['device']}; "
                  f"{ev['n_train']} train / {ev['n_test']} held-out sums")
        elif ev["event"] == "epoch":
            print(f"epoch {ev['epoch']:3d}  loss {ev['loss']:.4f}  "
                  f"held-out exact-match {ev['acc']:.1%}")
        elif ev["event"] == "checkpoint":
            print(f"saved {OUT / ev['name']}  ({ev['note']})")
        elif ev["event"] == "done":
            print(f"done: {ev['acc']:.1%} after {ev['epochs_run']} epochs")


if __name__ == "__main__":
    main()
