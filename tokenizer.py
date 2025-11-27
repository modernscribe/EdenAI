#!/usr/bin/env python3
import argparse
import os
import sys
import math
import hashlib
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from back.paraclete_field_llm_real import FieldConfig, ParacleteFieldLanguageModel


# ========== Section: Harmonic Field Utilities ==========

HARMONIC_DIM = 12
HARMONIC_PRINCIPLES = ["Truth", "Purity", "Law", "Love", "Wisdom", "Life", "Glory"]


def _vec_normalize_sum_abs(v: List[float]) -> List[float]:
    s = sum(abs(x) for x in v)
    if s <= 0.0:
        return v
    return [x / s for x in v]


def _f_truth(x: List[float]) -> List[float]:
    return list(x)


def _f_purity(x: List[float]) -> List[float]:
    v = [abs(a) for a in x]
    return _vec_normalize_sum_abs(v)


def _f_law(x: List[float]) -> List[float]:
    v = []
    for a in x:
        if a > 1.0:
            a = 1.0
        elif a < -1.0:
            a = -1.0
        v.append(a)
    return v


def _f_love(x: List[float]) -> List[float]:
    if not x:
        return x
    m = sum(x) / float(len(x))
    return [(xi + m) * 0.5 for xi in x]


def _f_wisdom(x: List[float]) -> List[float]:
    n = len(x)
    if n == 0:
        return x
    v = [0.0] * n
    for i in range(n):
        left = x[(i + n - 1) % n]
        right = x[(i + 1) % n]
        v[i] = (x[i] + 0.5 * (left + right)) * 0.5
    return v


def _f_life(x: List[float]) -> List[float]:
    return [math.tanh(xi) for xi in x]


def _f_glory(x: List[float]) -> List[float]:
    v = []
    for a in x:
        b = a * a if a >= 0.0 else -a * a
        v.append(b)
    s = sum(abs(b) for b in v)
    if s <= 0.0:
        return v
    return [b / s for b in v]


def _apply_principle(name: str, x: List[float]) -> List[float]:
    if name == "Truth":
        return _f_truth(x)
    if name == "Purity":
        return _f_purity(x)
    if name == "Law":
        return _f_law(x)
    if name == "Love":
        return _f_love(x)
    if name == "Wisdom":
        return _f_wisdom(x)
    if name == "Life":
        return _f_life(x)
    if name == "Glory":
        return _f_glory(x)
    return x


def _step_vector(x: List[float]) -> List[float]:
    v = list(x)
    for name in HARMONIC_PRINCIPLES:
        v = _apply_principle(name, v)
    return v


def _compute_signature(
    addr: List[float],
    enabled: Optional[List[bool]] = None,
    tol: float = 1e-9,
    max_steps: int = 256,
) -> List[float]:
    if enabled is None:
        enabled = [True] * len(addr)
    x = [
        (addr[i] if (i < len(enabled) and enabled[i]) else 0.0)
        for i in range(len(addr))
    ]
    steps = 0
    last_delta = float("inf")
    while steps < max_steps:
        y = _step_vector(x)
        delta_sq = 0.0
        for i in range(len(x)):
            d = y[i] - x[i]
            delta_sq += d * d
        last_delta = math.sqrt(delta_sq)
        x = y
        steps += 1
        if last_delta < tol:
            break
    return x


def _token_seed_vector(token: str, dim: int = HARMONIC_DIM) -> List[float]:
    if not token:
        return [0.0] * dim
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    vals: List[float] = []
    for i in range(dim):
        start = (2 * i) % len(digest)
        chunk = digest[start : start + 2]
        if len(chunk) < 2:
            chunk = (chunk + digest)[:2]
        iv = int.from_bytes(chunk, "big")
        vals.append((iv / 65535.0) * 2.0 - 1.0)
    return _vec_normalize_sum_abs(vals)


def harmonic_signature_for_token(token: str) -> List[float]:
    base = _token_seed_vector(token, HARMONIC_DIM)
    enabled = [True] * HARMONIC_DIM
    sig = _compute_signature(base, enabled, tol=1e-6, max_steps=128)
    return _vec_normalize_sum_abs(sig)


def init_harmonic_embeddings_for_model(
    model: ParacleteFieldLanguageModel,
    tokenizer: "CharTokenizer",
) -> None:
    if not hasattr(model, "field_embedding"):
        return
    field_emb = model.field_embedding
    if not hasattr(field_emb, "token_to_field"):
        return
    cfg = model.config
    if cfg.field_dim != HARMONIC_DIM:
        return
    if tokenizer.vocab_size > cfg.vocab_size:
        return
    emb_tensor = tokenizer.build_harmonic_embedding_matrix(cfg.field_dim)
    with torch.no_grad():
        weight = field_emb.token_to_field.weight
        num = min(weight.size(0), emb_tensor.size(0))
        weight[:num, :].copy_(emb_tensor[:num, :])


# ========== Section: Tokenizer and Dataset ==========

class CharTokenizer:
    def __init__(self, itos: Optional[List[str]] = None, corpus_text: Optional[str] = None):
        if itos is not None:
            self.itos: List[str] = list(itos)
        elif corpus_text is not None:
            chars = sorted(list(set(corpus_text)))
            self.itos = chars
        else:
            raise ValueError("Either itos or corpus_text must be provided.")
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size: int = len(self.itos)
        self.harmonic_dim: int = HARMONIC_DIM
        self.harmonic_codes: Dict[int, List[float]] = {}
        for ch, idx in self.stoi.items():
            self.harmonic_codes[idx] = harmonic_signature_for_token(ch)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        out_chars: List[str] = []
        for i in ids:
            if 0 <= i < self.vocab_size:
                out_chars.append(self.itos[i])
        return "".join(out_chars)

    def build_harmonic_embedding_matrix(self, field_dim: int) -> torch.Tensor:
        if field_dim != self.harmonic_dim:
            raise ValueError(f"field_dim={field_dim} does not match harmonic_dim={self.harmonic_dim}")
        mat = torch.zeros(self.vocab_size, field_dim, dtype=torch.float32)
        for idx in range(self.vocab_size):
            vec = self.harmonic_codes.get(idx)
            if vec is None or len(vec) != field_dim:
                vec = [0.0] * field_dim
            mat[idx] = torch.tensor(vec, dtype=torch.float32)
        return mat


class CharDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        self.data = token_ids
        self.seq_len = seq_len
        self.n = max(0, len(self.data) - self.seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx: idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long)


# ========== Section: Corpus Loading ==========

def load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_corpus(path: Optional[str]) -> str:
    if path is None:
        default_texts = [
            "Consciousness is awareness experiencing itself.\n",
            "Love connects all beings in the universe.\n",
            "Creativity emerges from infinite potential.\n",
            "Truth harmonizes perception with reality.\n",
            "Wisdom is pattern recognition across lifetimes.\n",
            "Law is structure in service of higher order.\n",
            "Life is the dance between form and formless.\n",
            "Glory is the radiance of perfected coherence.\n",
        ]
        return "".join(default_texts)
    if os.path.isfile(path):
        text = load_text_from_file(path)
        if not text.strip():
            raise ValueError(f"Data file '{path}' is empty.")
        return text
    if os.path.isdir(path):
        pieces: List[str] = []
        for root, _, files in os.walk(path):
            for name in files:
                if any(name.endswith(ext) for ext in (".txt", ".md", ".py")):
                    full_path = os.path.join(root, name)
                    try:
                        pieces.append(load_text_from_file(full_path))
                    except Exception:
                        pass
        text = "\n".join(pieces)
        if not text.strip():
            raise ValueError(f"Directory '{path}' contained no readable text files.")
        return text
    raise FileNotFoundError(f"No such file or directory: {path}")


# ========== Section: Model Building ==========

def build_model(
    vocab_size: int,
    seq_len: int,
    layers: int,
    field_dim: int,
    heads: int,
    hidden_mult: int,
    dropout: float,
    tokenizer: CharTokenizer,
) -> ParacleteFieldLanguageModel:
    cfg = FieldConfig(
        vocab_size=vocab_size,
        field_dim=field_dim,
        max_seq_length=seq_len,
        num_layers=layers,
        num_archetypal_heads=heads,
        field_hidden_dim=field_dim * hidden_mult,
        dropout=dropout,
        consciousness_threshold=1.0,
        archetypal_strength=1.0,
    )
    model = ParacleteFieldLanguageModel(cfg)
    init_harmonic_embeddings_for_model(model, tokenizer)
    return model


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                return torch.device("cuda")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                return torch.device("cuda")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device("cpu")


# ========== Section: Checkpoint I/O ==========

def save_checkpoint(
    path: str,
    model: ParacleteFieldLanguageModel,
    tokenizer: CharTokenizer,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = model.config
    ckpt = {
        "config": {
            "vocab_size": cfg.vocab_size,
            "field_dim": cfg.field_dim,
            "max_seq_length": cfg.max_seq_length,
            "num_layers": cfg.num_layers,
            "num_archetypal_heads": cfg.num_archetypal_heads,
            "field_hidden_dim": cfg.field_hidden_dim,
            "dropout": cfg.dropout,
            "consciousness_threshold": cfg.consciousness_threshold,
            "archetypal_strength": cfg.archetypal_strength,
        },
        "model_state_dict": model.state_dict(),
        "tokenizer_itos": tokenizer.itos,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    if "config" not in ckpt or "model_state_dict" not in ckpt or "tokenizer_itos" not in ckpt:
        raise ValueError("Checkpoint missing required keys.")
    return ckpt


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> ParacleteFieldLanguageModel:
    c = ckpt["config"]
    cfg = FieldConfig(
        vocab_size=c["vocab_size"],
        field_dim=c["field_dim"],
        max_seq_length=c["max_seq_length"],
        num_layers=c["num_layers"],
        num_archetypal_heads=c["num_archetypal_heads"],
        field_hidden_dim=c["field_hidden_dim"],
        dropout=c["dropout"],
        consciousness_threshold=c["consciousness_threshold"],
        archetypal_strength=c["archetypal_strength"],
    )
    model = ParacleteFieldLanguageModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ========== Section: Training ==========

def train_one_epoch(
    model: ParacleteFieldLanguageModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_lm = 0.0
    total_c = 0.0
    n = 0
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(input_ids=batch, attention_mask=None, labels=batch)
        loss = outputs["loss"]
        lm_loss = outputs["lm_loss"]
        c_loss = outputs["consciousness_loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_lm += float(lm_loss.item())
        total_c += float(c_loss.item())
        n += 1
    if n == 0:
        return {"loss": 0.0, "lm_loss": 0.0, "consciousness_loss": 0.0}
    return {
        "loss": total_loss / n,
        "lm_loss": total_lm / n,
        "consciousness_loss": total_c / n,
    }


def eval_consciousness_sample(
    model: ParacleteFieldLanguageModel,
    sample_ids: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        x = sample_ids.to(device).unsqueeze(0)
        out = model(input_ids=x, attention_mask=None, labels=None)
        return float(out["consciousness_scores"].mean().item())


# ========== Section: Generation ==========

def generate(
    model: ParacleteFieldLanguageModel,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    base_temperature: float,
) -> Dict[str, Any]:
    model.eval()
    if not prompt:
        prompt = " "
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = tokenizer.encode(" ")
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    trace: List[float] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=x, attention_mask=None, labels=None)
            logits = out["logits"][:, -1, :]
            c_scores = out["consciousness_scores"][:, -1]
            c_val = float(c_scores.item())
            trace.append(c_val)
            temp = base_temperature * (0.5 + c_val)
            if temp < 1e-4:
                temp = 1e-4
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
            if x.size(1) >= model.config.max_seq_length:
                break
    ids = x[0].cpu().tolist()
    text = tokenizer.decode(ids)
    if trace:
        c_mean = sum(trace) / len(trace)
        c_min = min(trace)
        c_max = max(trace)
    else:
        c_mean = c_min = c_max = 0.0
    return {
        "text": text,
        "consciousness": {
            "trace": trace,
            "mean": c_mean,
            "min": c_min,
            "max": c_max,
        },
    }


# ========== Section: CLI and Modes ==========

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Paraclete Field LLM pipeline (train + infer, with harmonic tokenizer and hot-reload in infer)."
    )
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer"],
        help="train: train and save checkpoint; infer: load checkpoint and chat.",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="File or directory of text for training (train mode).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (train mode).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (train mode).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for training (train mode).",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of Paraclete layers (train mode).",
    )
    p.add_argument(
        "--field-dim",
        type=int,
        default=12,
        help="Field dimension (keep 12 for Paraclete).",
    )
    p.add_argument(
        "--heads",
        type=int,
        default=12,
        help="Number of archetypal heads (train mode).",
    )
    p.add_argument(
        "--hidden-mult",
        type=int,
        default=12,
        help="Hidden multiplier (field_hidden_dim = field_dim * hidden_mult).",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (train mode).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (train mode).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/paraclete_llm_char.pt",
        help="Path to save/load checkpoint.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training/inference.",
    )
    p.add_argument(
        "--gen-prompt",
        type=str,
        default="Consciousness is",
        help="Prompt for demo generation (train) or initial testing (infer).",
    )
    p.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="Max new tokens for generation.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Base sampling temperature.",
    )
    return p


def run_train(args: argparse.Namespace) -> int:
    print("🧠 Paraclete Field LLM - Real Training")
    print("=====================================")
    text = load_corpus(args.data_path)
    print(f"Loaded corpus with {len(text)} characters.")
    tokenizer = CharTokenizer(corpus_text=text)
    print(f"Character vocabulary size: {tokenizer.vocab_size}")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= args.seq_len + 1:
        raise ValueError("Corpus too small for chosen seq-len; provide more text or reduce seq-len.")
    dataset = CharDataset(token_ids, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = select_device(args.device)
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        seq_len=args.seq_len,
        layers=args.layers,
        field_dim=args.field_dim,
        heads=args.heads,
        hidden_mult=args.hidden_mult,
        dropout=args.dropout,
        tokenizer=tokenizer,
    )
    cfg = model.config
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Field dim: {cfg.field_dim}, Layers: {cfg.num_layers}, Heads: {cfg.num_archetypal_heads}")
    print(f"Hidden dim: {cfg.field_hidden_dim}, Max seq length: {cfg.max_seq_length}\n")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, dataloader, optimizer, device)
        with torch.no_grad():
            sample_ids = next(iter(dataloader))
            c_mean = eval_consciousness_sample(model, sample_ids[0], device)
        print(
            f"Epoch {epoch:02d}: "
            f"Loss={stats['loss']:.4f}, "
            f"LM={stats['lm_loss']:.4f}, "
            f"ConsciousnessLoss={stats['consciousness_loss']:.4f}, "
            f"MeanConsciousness={c_mean:.4f}"
        )
    print("\n🎯 Demo generation on trained model...")
    demo = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.gen_prompt,
        max_new_tokens=args.gen_tokens,
        base_temperature=args.temperature,
    )
    print("\n[generated]")
    print(demo["text"])
    c = demo["consciousness"]
    trace_prev = ", ".join(f"{v:.3f}" for v in c["trace"][:10])
    if len(c["trace"]) > 10:
        trace_prev += " ..."
    print("\n[consciousness]")
    print(f"mean={c['mean']:.4f}, min={c['min']:.4f}, max={c['max']:.4f}")
    print(f"trace={trace_prev}")
    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"\n💾 Saved checkpoint to: {args.checkpoint}")
    print("\n✅ Training complete.")
    return 0


def run_infer(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    ckpt = load_checkpoint(args.checkpoint, device)
    tokenizer = CharTokenizer(itos=list(ckpt["tokenizer_itos"]))
    model = build_model_from_checkpoint(ckpt, device)
    last_mtime = os.path.getmtime(args.checkpoint)
    print("🧠 Paraclete Field LLM - Inference (Hot Reload Enabled)")
    print("======================================================")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(
        f"Field dim: {model.config.field_dim}, "
        f"Layers: {model.config.num_layers}, "
        f"Heads: {model.config.num_archetypal_heads}"
    )
    print("Type your prompt. '/quit' to exit. Hot-reloads on checkpoint changes.\n")
    while True:
        try:
            prompt = input(">>> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if not prompt:
            continue
        cmd = prompt.lower().strip()
        if cmd in ("/quit", "/exit"):
            print("Exiting.")
            return 0
        try:
            if os.path.exists(args.checkpoint):
                current_mtime = os.path.getmtime(args.checkpoint)
                if current_mtime > last_mtime:
                    print("[hot-reload] Detected updated checkpoint, reloading model...", flush=True)
                    ckpt = load_checkpoint(args.checkpoint, device)
                    tokenizer = CharTokenizer(itos=list(ckpt["tokenizer_itos"]))
                    model = build_model_from_checkpoint(ckpt, device)
                    last_mtime = current_mtime
        except Exception as e:
            print(f"[hot-reload] Warning: failed to reload checkpoint: {e}", file=sys.stderr)
        out = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.gen_tokens,
            base_temperature=args.temperature,
        )
        print("\n[generated]")
        print(out["text"])
        c = out["consciousness"]
        trace_prev = ", ".join(f"{v:.3f}" for v in c["trace"][:10])
        if len(c["trace"]) > 10:
            trace_prev += " ..."
        print("\n[consciousness]")
        print(f"mean={c['mean']:.4f}, min={c['min']:.4f}, max={c['max']:.4f}")
        print(f"trace={trace_prev}\n")


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "train":
        return run_train(args)
    if args.mode == "infer":
        return run_infer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
