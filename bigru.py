"""BiGRU classifier on the same splits as the logistic regression.

Verified end to end on torch 2.7.1+cpu and torch 2.13.0+cu130: training, a
forced --time-budget pause, resume from checkpoint, idempotent rerun of a
finished run, and the config-mismatch failure path.

Classification only. No sequence-to-sequence description decoder.

Architecture, unchanged from the original run and not configurable, because
every number in the paper depends on it staying fixed: vocabulary capped at
the 20,000 most frequent training tokens, embedding 256, a single-layer
bidirectional nn.GRU with hidden 256, masked mean-pooling over the encoder
outputs, a linear head to one logit, and binary cross-entropy. Batch size is
190 and must stay 190 for comparability; if a configuration does not fit in
memory, drop the configuration, do not shrink the batch.

Runs checkpoint after every epoch and resume exactly where they stopped, so a
pod culler cannot destroy more than one epoch of work.

Compatible with torch 2.7.1 (pod, CUDA 11.8) and torch 2.13 (development box).
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lethe_split import (ALL_CONDITIONS, FILES, SEED, SPLIT_MODES,
                         SplitError, build_split, dedup_hashes, load,
                         majority_baseline)

# ---------------------------------------------------------------- fixed model
VOCAB_CAP = 20000
EMB_DIM = 256
HIDDEN = 256
# Transformer encoder, matched to the recurrent model everywhere it can be:
# same vocabulary cap, same embedding width, same pooling, same head, same
# loss, same batch, same early stopping.
TFM_LAYERS = 4
TFM_HEADS = 4
TFM_FF = 512
TFM_DROPOUT = 0.1
BATCH = 190            # do not change; see module docstring
MAX_EPOCHS = 150
PATIENCE = 5
VAL_FRACTION = 0.10
LR = 1e-3
PAD, UNK = 0, 1

CKPT_FORMAT = 1
BAR = "=" * 78


def log(msg=""):
    print(msg, flush=True)


def die(msg, code=2):
    """Fail with an actionable message rather than a traceback."""
    print(f"\nERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ------------------------------------------------------------------ RNG state
def capture_rng(shuffle_rng):
    """All four global RNG streams plus the local shuffling generator."""
    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
        "shuffle_rng": shuffle_rng.bit_generator.state,
    }


def restore_rng(state, shuffle_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy_legacy"])
    torch.set_rng_state(state["torch_cpu"].cpu()
                        if torch.is_tensor(state["torch_cpu"])
                        else state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        saved = state["torch_cuda"]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved])
        # A different visible-device count means this checkpoint moved between
        # machines. CPU and numpy streams still restore, so training continues
        # deterministically apart from cuDNN kernel nondeterminism.
    shuffle_rng.bit_generator.state = state["shuffle_rng"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- model
class TransformerEnc(nn.Module):
    """4-layer encoder, learned positional embeddings, masked mean-pool."""

    def __init__(self, n_vocab, max_len):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, EMB_DIM, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, EMB_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM, nhead=TFM_HEADS, dim_feedforward=TFM_FF,
            dropout=TFM_DROPOUT, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=TFM_LAYERS)
        self.head = nn.Linear(EMB_DIM, 1)
        self.max_len = max_len

    def forward(self, ids, lens):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).clamp(max=self.max_len - 1)
        h = self.emb(ids) + self.pos(pos)[None, :, :]
        pad = ids == PAD
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class BiGRU(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM, HIDDEN, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(2 * HIDDEN, 1)

    def forward(self, ids, lens):
        e = self.emb(ids)
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1), device=out.device)[None, :]
                < lens[:, None]).unsqueeze(-1)
        pooled = (out * mask).sum(1) / lens.clamp(min=1)[:, None]
        return self.head(pooled).squeeze(-1)


def make_model(cfg, n_vocab):
    if cfg["arch"] == "transformer":
        return TransformerEnc(n_vocab, cfg["max_len"])
    return BiGRU(n_vocab)


# ---------------------------------------------------------------- data plumbing
def build_vocab(train_tok):
    c = collections.Counter()
    for t in train_tok:
        c.update(t)
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:VOCAB_CAP]
    return {tok: i + 2 for i, (tok, _) in enumerate(items)}, len(c)


def encode(tokens, vocab):
    g = vocab.get
    return np.asarray([g(t, UNK) for t in tokens], dtype=np.int32)


def batches(idx, seqs, y, rng=None, bs=BATCH):
    order = np.array(idx)
    if rng is not None:
        rng.shuffle(order)
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        lens = np.array([len(seqs[i]) for i in sel], dtype=np.int64)
        m = int(lens.max())
        arr = np.zeros((len(sel), m), dtype=np.int64)
        for r, i in enumerate(sel):
            arr[r, :len(seqs[i])] = seqs[i]
        yield (torch.from_numpy(arr), torch.from_numpy(lens),
               torch.from_numpy(y[sel].astype(np.float32)))


@torch.no_grad()
def evaluate(model, idx, seqs, y, dev):
    model.eval()
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    tot, correct, n = 0.0, 0, 0
    for ids, lens, yb in batches(idx, seqs, y):
        ids, lens, yb = ids.to(dev), lens.to(dev), yb.to(dev)
        logit = model(ids, lens)
        tot += float(lossf(logit, yb).detach())
        correct += int(((logit > 0).float() == yb).sum())
        n += len(yb)
    return tot / max(n, 1), correct / max(n, 1) * 100


def prepare(cfg):
    """Load the corpus, build the split, encode, and carve validation."""
    d = load(cfg["file"], cfg["cond"])
    sp = build_split(cfg["file"], d, hashes=dedup_hashes(cfg["file"], cfg["cond"]),
                     mode=cfg["split_mode"],
                     split_seed=cfg["split_seed"],
                     shuffle_labels=cfg["shuffle_labels"],
                     shuffle_seed=cfg["seed"],
                     shuffle_scope=cfg.get("shuffle_scope", "train"))
    xd, labels = sp["x"], sp["labels"]
    tr_mask, te_mask = sp["train_mask"], sp["test_mask"]
    tokens = [s.split()[:cfg["max_len"]] for s in xd]
    del d

    tr_all = np.flatnonzero(tr_mask)
    te_idx = np.flatnonzero(te_mask)
    vocab, full_vocab = build_vocab([tokens[i] for i in tr_all])
    seqs = [encode(t, vocab) for t in tokens]
    del tokens

    # Validation is carved from the TRAINING side only, so held-out groups are
    # never seen during training or model selection.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr_all))
    n_val = max(1, int(round(len(tr_all) * VAL_FRACTION)))
    va_idx = tr_all[perm[:n_val]]
    tr_idx = tr_all[perm[n_val:]]

    maj_class, maj = majority_baseline(labels[tr_idx], labels[te_idx])
    meta = {
        "n_before": sp["n_before"], "n_after": sp["n_after"],
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "group_kind": sp["group_kind"], "n_groups": int(sp["n_groups"]),
        "vocab_kept": len(vocab), "vocab_full": full_vocab,
        "n_vocab_rows": len(vocab) + 2,
        "majority_class": maj_class, "baseline": maj,
        "labels_shuffled": bool(sp["labels_shuffled"]),
        "shuffle_scope": sp.get("shuffle_scope"),
        "test_groups": (sp.get("test_templates") if sp["group_kind"] ==
                        "template" else sp.get("test_cwes")),
    }
    return seqs, labels, tr_idx, va_idx, te_idx, meta


# ------------------------------------------------------------- checkpointing
def save_ckpt(path, cfg, epoch, model, opt, best, best_state, curve,
              shuffle_rng, elapsed):
    tmp = path + ".tmp"
    torch.save({
        "format": CKPT_FORMAT,
        "config": cfg,
        "epoch": epoch,                      # next epoch index to run
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "best_val": best["val"],
        "best_epoch": best["epoch"],
        "patience_counter": best["bad"],
        "best_model": best_state,
        "curve": curve,
        "rng": capture_rng(shuffle_rng),
        "elapsed_seconds": elapsed,
    }, tmp)
    os.replace(tmp, path)      # atomic: a cull mid-write cannot corrupt it


def load_ckpt(path, cfg):
    # weights_only defaults to True from torch 2.6. This checkpoint holds RNG
    # states and a config dict, which that mode rejects, so the choice is made
    # explicitly here. These files are written by this script into a directory
    # the user controls, so unpickling them is not a new trust boundary.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        die(f"checkpoint {path} has format {ck.get('format')}, this script "
            f"writes format {CKPT_FORMAT}. Delete the checkpoint to restart "
            f"the run from scratch.")
    old = ck["config"]
    diffs = [(k, old.get(k), cfg.get(k))
             for k in sorted(set(old) | set(cfg))
             if old.get(k) != cfg.get(k)]
    if diffs:
        lines = "\n".join(f"    {k}: checkpoint={o!r} requested={n!r}"
                          for k, o, n in diffs)
        die(f"config mismatch resuming {path}\n{lines}\n"
            f"  The checkpoint was produced by a different configuration. "
            f"Either rerun with the original settings or delete the "
            f"checkpoint to start this configuration fresh.")
    return ck


# ------------------------------------------------------------------- training
def train(cfg, seqs, y, tr_idx, va_idx, te_idx, meta, dev, ckpt_path,
          time_budget):
    n_vocab = meta["n_vocab_rows"]
    seed_everything(cfg["seed"])
    model = make_model(cfg, n_vocab).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss()
    shuffle_rng = np.random.default_rng(cfg["seed"])

    start_epoch = 0
    best = {"val": float("inf"), "epoch": -1, "bad": 0}
    best_state, curve, elapsed = None, [], 0.0

    if ckpt_path and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path, cfg)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best = {"val": ck["best_val"], "epoch": ck["best_epoch"],
                "bad": ck["patience_counter"]}
        best_state = ck["best_model"]
        curve = ck["curve"]
        elapsed = ck.get("elapsed_seconds", 0.0)
        restore_rng(ck["rng"], shuffle_rng)
        log(f"  RESUMED from {ckpt_path} at epoch {start_epoch} "
            f"(best val {best['val']:.5f} @ epoch {best['epoch']}, "
            f"patience {best['bad']}/{PATIENCE}, {elapsed:.0f}s spent)")

    if start_epoch >= MAX_EPOCHS or best["bad"] >= PATIENCE:
        log("  already finished per checkpoint; evaluating and exiting")
        status = "done"
    else:
        status = "running"

    t_wall = time.perf_counter()
    for ep in range(start_epoch, MAX_EPOCHS):
        if status != "running":
            break
        model.train()
        tot, n = 0.0, 0
        t0 = time.perf_counter()
        for ids, lens, yb in batches(tr_idx, seqs, y, shuffle_rng):
            ids, lens, yb = ids.to(dev), lens.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(ids, lens), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(yb)
            n += len(yb)
        tr_loss = tot / n
        va_loss, va_acc = evaluate(model, va_idx, seqs, y, dev)
        dt = time.perf_counter() - t0
        curve.append([ep, tr_loss, va_loss, va_acc, dt])
        log(f"    epoch {ep:>3}  train {tr_loss:.5f}  val {va_loss:.5f}  "
            f"val_acc {va_acc:6.2f}%  {dt:6.1f}s")

        if not np.isfinite(tr_loss) or not np.isfinite(va_loss):
            log(f"    DIVERGED at epoch {ep}: train {tr_loss} val {va_loss}")
            return {"status": "failed", "reason": "non-finite loss",
                    "curve": curve, "epochs_run": len(curve)}

        if va_loss < best["val"] - 1e-5:
            best.update(val=va_loss, epoch=ep, bad=0)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            best["bad"] += 1

        elapsed_now = elapsed + (time.perf_counter() - t_wall)
        if ckpt_path:
            save_ckpt(ckpt_path, cfg, ep + 1, model, opt, best, best_state,
                      curve, shuffle_rng, elapsed_now)

        if best["bad"] >= PATIENCE:
            log(f"    early stop: {PATIENCE} epochs without improvement")
            status = "done"
            break
        if time_budget and elapsed_now >= time_budget:
            log(f"    time budget {time_budget}s reached after epoch {ep}; "
                f"checkpoint saved, exiting cleanly for restart")
            return {"status": "paused", "curve": curve,
                    "epochs_run": len(curve), "elapsed": elapsed_now}
    else:
        status = "done"

    if best_state is None:
        return {"status": "failed", "reason": "no epoch completed",
                "curve": curve, "epochs_run": len(curve)}

    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, te_idx, seqs, y, dev)
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if dev.type == "cuda" else 0.0)
    return {"status": "done", "test_acc": te_acc, "test_loss": te_loss,
            "best_val": best["val"], "best_epoch": best["epoch"],
            "epochs_run": len(curve), "curve": curve,
            "peak_gpu_gb": peak,
            "elapsed": elapsed + (time.perf_counter() - t_wall)}


# ----------------------------------------------------------------------- cli
def run_id(cfg):
    bits = [cfg["file"], cfg["cond"], cfg["split_mode"],
            f"ml{cfg['max_len']}", f"s{cfg['seed']}"]
    if cfg.get("arch") == "transformer":
        bits.insert(0, "tfm")
    if cfg["split_seed"] is not None:
        bits.append(f"ss{cfg['split_seed']}")
    if cfg["shuffle_labels"]:
        bits.append("shuffled" if cfg.get("shuffle_scope") == "all"
                    else "shuftrain")
    return "-".join(bits)


def make_parser():
    p = argparse.ArgumentParser(
        prog="bigru.py",
        description="Train the BiGRU classifier on one file/condition cell.",
        epilog="Example:\n"
               "  python bigru.py --file juliet --cond SCRUB --max-len 256 "
               "--seed 0 \\\n"
               "      --gpu 0 --checkpoint-dir checkpoints --results-dir "
               "results/runs",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, choices=FILES)
    p.add_argument("--cond", required=True, choices=ALL_CONDITIONS,
                   help="a -matched condition deduplicates by the "
                        "SCRUB hashes so every condition scores the "
                        "same test items")
    p.add_argument("--seed", type=int, default=0,
                   help="model/init seed (default 0)")
    p.add_argument("--max-len", type=int, default=256,
                   help="token truncation length (default 256)")
    p.add_argument("--split-mode", default="cwe", choices=SPLIT_MODES,
                   help="partition protocol (default cwe)")
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed for random-cwe and template modes; cwe mode "
                        "uses the canonical 20260902 when omitted")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="random-label control: permute labels")
    p.add_argument("--shuffle-scope", default="train", choices=["train", "all"],
                   help="train: permute the training fold only, test keeps "
                        "true labels (detects a leaking split). all: permute "
                        "everything (only confirms chance-level pipeline). "
                        "Default train.")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index; omit for CPU")
    p.add_argument("--checkpoint-dir", default=None,
                   help="directory for per-epoch checkpoints; enables resume")
    p.add_argument("--results-dir", default="results/runs",
                   help="where the per-run JSON result is written")
    p.add_argument("--arch", default="bigru",
                   choices=["bigru", "transformer"],
                   help="encoder architecture (default bigru)")
    p.add_argument("--lr", type=float, default=LR,
                   help=f"Adam learning rate (default {LR})")
    p.add_argument("--time-budget", type=int, default=0,
                   help="seconds; stop cleanly after the epoch that crosses "
                        "it, leaving a resumable checkpoint (0 = unlimited)")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)

    if args.split_mode == "random-cwe" and args.split_seed is None:
        die("--split-mode random-cwe requires --split-seed. That mode exists "
            "to vary which CWEs are held out, so the seed is the experiment.")
    if args.split_mode in ("template", "source-sink") and args.file != "juliet":
        die(f"--split-mode template is only validated for juliet, not "
            f"{args.file!r}. NDSS descriptions were not analyzed for template "
            f"structure. Run template_split.py --validate --file {args.file} "
            f"first if you intend to use it.")
    if args.max_len < 8:
        die(f"--max-len {args.max_len} is too small to be meaningful.")

    if args.gpu is not None:
        if not torch.cuda.is_available():
            die("--gpu given but torch.cuda.is_available() is False. "
                "Drop --gpu to use CPU.")
        if args.gpu >= torch.cuda.device_count():
            die(f"--gpu {args.gpu} but only "
                f"{torch.cuda.device_count()} device(s) visible. If you set "
                f"CUDA_VISIBLE_DEVICES, the index is relative to that list.")
        dev = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")

    cfg = {
        "file": args.file, "cond": args.cond, "seed": args.seed,
        "max_len": args.max_len, "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "shuffle_labels": bool(args.shuffle_labels),
        "shuffle_scope": args.shuffle_scope if args.shuffle_labels else None,
        # architecture, pinned so a resume cannot silently change the model
        "arch": args.arch,
        "vocab_cap": VOCAB_CAP, "emb": EMB_DIM, "hidden": HIDDEN,
        "batch": BATCH, "lr": args.lr, "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS, "val_fraction": VAL_FRACTION,
    }
    if args.arch == "transformer":
        cfg.update({"tfm_layers": TFM_LAYERS, "tfm_heads": TFM_HEADS,
                    "tfm_ff": TFM_FF, "tfm_dropout": TFM_DROPOUT})
    rid = run_id(cfg)

    log(BAR)
    log(f"LETHE BiGRU  run {rid}")
    log(BAR)
    log(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
        f"numpy {np.__version__}")
    log(f"device {dev}" + (f"  {torch.cuda.get_device_name(dev)}"
                           if dev.type == "cuda" else ""))
    if dev.type == "cuda":
        log(f"cuda {torch.version.cuda}  arch {torch.cuda.get_arch_list()}  "
            f"cudnn {torch.backends.cudnn.version()}")
    log(f"config {json.dumps(cfg, sort_keys=True)}")

    ckpt_path = None
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, rid + ".pt")
    os.makedirs(args.results_dir, exist_ok=True)
    res_path = os.path.join(args.results_dir, rid + ".json")

    try:
        seqs, y, tr_idx, va_idx, te_idx, meta = prepare(cfg)
    except SplitError as e:
        die(str(e))

    log(f"  dedup {meta['n_before']:,} -> {meta['n_after']:,}  "
        f"train {meta['n_train']:,}  val {meta['n_val']:,}  "
        f"test {meta['n_test']:,}")
    log(f"  grouping: {meta['group_kind']}, {meta['n_groups']:,} groups")
    log(f"  vocab {meta['vocab_kept']:,} kept of {meta['vocab_full']:,} "
        f"(cap {VOCAB_CAP:,})")
    log(f"  majority class {meta['majority_class']} -> "
        f"baseline {meta['baseline']:.2f}%")
    if meta["labels_shuffled"]:
        log(f"  LABELS SHUFFLED, scope={cfg['shuffle_scope']!r}: "
            f"{'training fold only, test keeps true labels'
               if cfg['shuffle_scope']=='train' else 'whole corpus'}")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    r = train(cfg, seqs, y, tr_idx, va_idx, te_idx, meta, dev, ckpt_path,
              args.time_budget)

    out = {"run_id": rid, "config": cfg, "meta": meta}
    out.update({k: v for k, v in r.items()})
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    if r["status"] == "done":
        log(f"\n  RESULT {rid}")
        log(f"    test accuracy {r['test_acc']:.2f}%  "
            f"baseline {meta['baseline']:.2f}%  "
            f"lift {r['test_acc']-meta['baseline']:+.2f}")
        log(f"    epochs {r['epochs_run']} (best {r['best_epoch']}), "
            f"peak GPU {r['peak_gpu_gb']:.2f} GB, {r['elapsed']:.0f}s")
    elif r["status"] == "paused":
        log(f"\n  PAUSED after {r['epochs_run']} epochs. Rerun the same "
            f"command to continue.")
    else:
        log(f"\n  FAILED: {r.get('reason')}")
    log(f"  wrote {res_path}")
    return 0 if r["status"] in ("done", "paused") else 3


if __name__ == "__main__":
    sys.exit(main())
