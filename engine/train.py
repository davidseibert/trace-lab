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
gets a training-time axis over a *real* transformers model.

Answers are fixed-width, most significant digit first: prompt "17+25=" and the
model should answer '0', '4', '2'.

    make train                 # Docker, GPU (EPOCHS=... to override)
    uv run python train.py     # bare metal — NOTE: saves to engine/data/local,
                               # a different filesystem than the Docker volume;
                               # train where you serve.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

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


def save(model, tokenizer, name: str, note: str) -> None:
    path = OUT / name
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"saved {path}  ({note})")


def main() -> None:
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
    print(f"{n_params / 1e6:.2f}M params on {device}; {len(train)} train / {len(test)} held-out sums")

    save(model, tokenizer, "add-step0", "untrained — the J-lens ladder should be garbage")

    labels = train.clone()
    labels[:, :PROMPT_LEN] = -100  # loss only on the answer digits
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    mid_saved = False
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(train), device=device)
        total = 0.0
        for i in range(0, len(train), BATCH):
            idx = perm[i : i + BATCH]
            out = model(input_ids=train[idx], labels=labels[idx])
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(out.loss) * len(idx)
        acc = exact_match(model, test)
        print(f"epoch {epoch:3d}  loss {total / len(train):.4f}  held-out exact-match {acc:.1%}")
        if not mid_saved and acc >= 0.5:
            save(model, tokenizer, "add-mid", f"first ≥50% checkpoint (epoch {epoch}, {acc:.1%})")
            mid_saved = True
        if acc >= 0.999:
            print("early stop: ≥99.9% held-out")
            break

    if not mid_saved:
        save(model, tokenizer, "add-mid", "never crossed 50% — saved final state as mid")
    save(model, tokenizer, "add-final", f"final ({acc:.1%} held-out exact-match)")


if __name__ == "__main__":
    main()
