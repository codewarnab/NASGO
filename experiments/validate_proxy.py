#!/usr/bin/env python3
"""Bounded validation harness: does NASGO's zero-cost proxy fitness predict
real trained validation accuracy on CIFAR-10?

Subcommands:
  select   - choose architectures from NASGO SQLite experiment DBs
  run      - train pending architectures incrementally (resumable, budget-limited)
  analyze  - Spearman correlation + bootstrap CI from results.jsonl

Reuses the repo's model construction (scripts/train.py) so measured
architectures are exactly what NASGO would train.
"""
import argparse, json, os, random, sqlite3, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def import_repo_train(repo):
    sys.path.insert(0, str(Path(repo) / "scripts"))
    import train as repo_train  # noqa
    return repo_train


# ---------- dataset ----------
class FolderCIFAR(torch.utils.data.Dataset):
    """CIFAR-10 in fast-ai folder layout (train/<class>/*.png, test/<class>/*.png)."""
    def __init__(self, root, split, indices=None, train_transform=False):
        cache = Path(root).parent / f".filelist-{split}.json"
        if cache.exists():
            raw = json.loads(cache.read_text())
            self.samples = [(Path(root) / rel, ci) for rel, ci in raw]
        else:
            self.samples = []
            for ci, cname in enumerate(CLASSES):
                d = Path(root) / split / cname
                files = sorted(p.name for p in d.glob("*.png"))
                for f in files:
                    self.samples.append((d / f, ci))
            cache.write_text(json.dumps(
                [[str(p.relative_to(root)), ci] for p, ci in self.samples]))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self.train_transform = train_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.train_transform:
            # random crop with padding 4 + horizontal flip (matches train.py)
            arr = np.pad(arr, ((4, 4), (4, 4), (0, 0)), mode="constant")
            rng = np.random.random
            top = int(rng() * 9); left = int(rng() * 9)
            arr = arr[top:top + 32, left:left + 32]
            if rng() < 0.5:
                arr = arr[:, ::-1]
        arr = (arr - np.array(CIFAR_MEAN, np.float32)) / np.array(CIFAR_STD, np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1)), label


def deterministic_indices(seed, n_total, n_take):
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    return idx[:n_take]


# ---------- baseline ----------
class BaselineCNN(nn.Module):
    """Plain 3-block CNN: a simple hand-designed reference point."""
    def __init__(self, channels=32, num_classes=10):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(2 * c, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


# ---------- select ----------
def cmd_select(args):
    selected = {}  # key: dedup hash -> record
    def canon(spec):
        n = spec["normal_cell"]; r = spec["reduction_cell"]
        return json.dumps({"normal_cell": n, "reduction_cell": r}, sort_keys=True)

    # stratified sample from random-strategy pool across full fitness range
    db = sqlite3.connect(args.random_db)
    rows = db.execute("SELECT arch_json, fitness FROM architectures ORDER BY fitness").fetchall()
    rows = sorted({canon(json.loads(a)): (a, f) for a, f in rows}.values(),
                  key=lambda t: t[1])
    n = len(rows)
    picks = sorted(set(int(round(i * (n - 1) / (args.n_stratified - 1)))
                       for i in range(args.n_stratified)))
    for rank in picks:
        a, f = rows[rank]
        spec = json.loads(a)
        selected[canon(spec)] = {"arch": {"normal_cell": spec["normal_cell"],
                                          "reduction_cell": spec["reduction_cell"]},
                                 "proxy_fitness": f, "origin": f"random_rank{rank}_of_{n}"}
    # best of each strategy pool
    for name, path in [("random", args.random_db), ("evolutionary", args.evolutionary_db),
                       ("regularized", args.regularized_db)]:
        db = sqlite3.connect(path)
        a, f = db.execute("SELECT arch_json, fitness FROM architectures "
                          "ORDER BY fitness DESC LIMIT 1").fetchone()
        spec = json.loads(a)
        selected.setdefault(canon(spec), {"arch": {"normal_cell": spec["normal_cell"],
                                                   "reduction_cell": spec["reduction_cell"]},
                                          "proxy_fitness": f, "origin": f"{name}_best"})
    out = list(selected.values())
    for i, rec in enumerate(out):
        rec["arch_id"] = f"arch-{i:02d}"
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"selected {len(out)} architectures -> {args.out}")
    for rec in out:
        print(f"  {rec['arch_id']} proxy={rec['proxy_fitness']:.4f} {rec['origin']}")


# ---------- run ----------
def train_one(model, train_ds, val_ds, epochs, lr, weight_decay, device,
              batch_size, run_seed, subset_seed, ckpt_path):
    """Train with per-epoch checkpointing so a preempted run resumes exactly.

    Shuffle order and augmentation randomness are reseeded per epoch from
    (run_seed, epoch), so a restarted epoch is deterministic.
    """
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, log, start_ep = 0.0, [], 0
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, log = st["best"], st["log"]
        start_ep = st["epoch"] if st.get("partial") else st["epoch"] + 1
        if st.get("partial"):
            log = log[:-1] if len(log) > st["epoch"] else log
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    for ep in range(start_ep, epochs):
        ep_seed = subset_seed * 100003 + run_seed * 101 + ep
        np.random.seed(ep_seed % (2 ** 32))
        gen = torch.Generator().manual_seed(ep_seed)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        if time.time() > getattr(train_one, "_deadline", float("inf")):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best, "log": log,
                        "epoch": ep, "partial": True}, ckpt_path)
            raise TimeoutError()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best = max(best, acc)
        log.append(round(acc, 4))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best": best, "log": log, "epoch": ep},
                   ckpt_path)
    ckpt_path.unlink(missing_ok=True)
    return best, log


def cmd_run(args):
    repo_train = import_repo_train(args.repo)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    train_idx = deterministic_indices(args.subset_seed, 50000, args.train_size)
    val_idx = deterministic_indices(args.subset_seed + 1, 10000, args.val_size)
    train_ds = FolderCIFAR(args.data, "train", train_idx, train_transform=True)
    val_ds = FolderCIFAR(args.data, "test", val_idx, train_transform=False)
    ckpt_dir = Path(args.results).parent / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    jobs = json.loads(Path(args.archs).read_text())
    # add baselines
    jobs.append({"arch_id": "baseline-cnn32", "kind": "baseline", "channels": 32,
                 "proxy_fitness": None, "origin": "hand_designed_baseline"})
    results_path = Path(args.results)
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            done.add(json.loads(line)["run_id"])

    deadline = time.time() + args.budget_seconds
    ran = 0
    for job in jobs:
        for seed in args.seeds:
            run_id = f"{job['arch_id']}-seed{seed}"
            if run_id in done:
                continue
            if time.time() > deadline:
                print(f"budget exhausted after {ran} runs; rerun to resume", file=sys.stderr)
                return
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            train_one._deadline = deadline
            t0 = time.time()
            if job.get("kind") == "baseline":
                model = BaselineCNN(channels=job["channels"])
            else:
                model = repo_train.NASNetwork(
                    {"normal_cell": job["arch"]["normal_cell"],
                     "reduction_cell": job["arch"]["reduction_cell"]},
                    channels=args.channels, layers=args.layers, num_classes=10)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            error = None
            try:
                best, log = train_one(model, train_ds, val_ds, args.epochs,
                                      args.lr, args.weight_decay, device,
                                      args.batch_size, seed, args.subset_seed,
                                      ckpt_dir / f"{run_id}.pt")
            except TimeoutError:
                print(f"{run_id}: epoch checkpointed, budget spent; rerun to resume", flush=True)
                return
            except Exception as e:  # record and continue; failures are evidence
                error = f"{type(e).__name__}: {e}"
                best, log = 0.0, []
            dt = time.time() - t0
            rec = {"run_id": run_id, "arch_id": job["arch_id"], "seed": seed,
                   "proxy_fitness": job.get("proxy_fitness"), "origin": job.get("origin"),
                   "kind": job.get("kind", "nas"),
                   "best_val_acc": round(best, 4), "val_acc_per_epoch": log,
                   "params": params, "train_seconds": round(dt, 1),
                   "error": error,
                   "config": {"train_size": args.train_size, "val_size": args.val_size,
                              "epochs": args.epochs, "batch_size": args.batch_size,
                              "channels": args.channels, "layers": args.layers,
                              "lr": args.lr, "weight_decay": args.weight_decay,
                              "subset_seed": args.subset_seed}}
            with results_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            ran += 1
            print(f"{run_id}: best_val={best:.4f} params={params} {dt:.0f}s {log} {error or ''}", flush=True)


# ---------- analyze ----------
def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(v))
        # average ties
        v = np.asarray(v, float); r = np.empty(len(v))
        sv = np.sort(v); idx = np.argsort(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[idx[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def cmd_analyze(args):
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines()]
    nas = {}
    for r in recs:
        if r["kind"] != "nas":
            continue
        nas.setdefault(r["arch_id"], {"proxy": r["proxy_fitness"], "accs": [], "params": r["params"]})
        nas[r["arch_id"]]["accs"].append(r["best_val_acc"])
    ids = sorted(nas)
    proxy = [nas[i]["proxy"] for i in ids]
    acc_mean = [float(np.mean(nas[i]["accs"])) for i in ids]
    rho = spearman(proxy, acc_mean)
    # bootstrap CI over architectures
    rng = np.random.default_rng(0)
    boots = []
    n = len(ids)
    for _ in range(10000):
        sel = rng.integers(0, n, n)
        if len(set(proxy[i] for i in sel)) < 2:
            continue
        boots.append(spearman([proxy[i] for i in sel], [acc_mean[i] for i in sel]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
    print(f"n={n} architectures, seeds per arch: {sorted({len(nas[i]['accs']) for i in ids})}")
    print(f"Spearman rho(proxy, trained_val_acc) = {rho:.3f}  bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"\n{'arch':10s} {'proxy':>7s} {'val_acc(mean)':>13s} {'accs':>16s} {'params':>8s}")
    for i in sorted(ids, key=lambda k: -np.mean(nas[k]['accs'])):
        print(f"{i:10s} {nas[i]['proxy']:7.4f} {np.mean(nas[i]['accs']):13.4f} "
              f"{str(nas[i]['accs']):>16s} {nas[i]['params']:8d}")
    base = [r for r in recs if r["kind"] == "baseline"]
    for r in base:
        print(f"baseline {r['arch_id']} seed{r['seed']}: best_val={r['best_val_acc']:.4f} params={r['params']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--random-db", required=True)
    s.add_argument("--evolutionary-db", required=True)
    s.add_argument("--regularized-db", required=True)
    s.add_argument("--n-stratified", type=int, default=12)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--archs", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--budget-seconds", type=float, default=100)
    r.add_argument("--train-size", type=int, default=4096)
    r.add_argument("--val-size", type=int, default=1024)
    r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--channels", type=int, default=16)
    r.add_argument("--layers", type=int, default=8)
    r.add_argument("--lr", type=float, default=0.025)
    r.add_argument("--weight-decay", type=float, default=3e-4)
    r.add_argument("--subset-seed", type=int, default=123)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    a = sub.add_parser("analyze")
    a.add_argument("--results", required=True)
    args = p.parse_args()
    {"select": cmd_select, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
