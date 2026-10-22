import argparse, importlib.util, json, random, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Locate this script's directory, rather than using a personal path.
SCRIPT_DIR = Path(__file__).resolve().parent

# Training script should be in the same directory.
TRAIN_SCRIPT = SCRIPT_DIR / "train_adni_rgaf_shmr.py"

# Default output folder.
DEFAULT_ROOT = SCRIPT_DIR / "results"


def load_training_module():
    spec = importlib.util.spec_from_file_location("r2c3_main_training", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_eligibility(df):
    # Exploratory-only gate. This is not final QC approval.
    if "exploratory_only" not in df.columns:
        raise ValueError("Exploratory run blocked: missing exploratory_only marker.")
    exploratory = df["exploratory_only"].astype(str).str.strip().str.lower()
    if not exploratory.eq("true").all():
        raise ValueError("Exploratory run blocked: every row must be marked exploratory_only=True.")
    if "final_qc_approval" not in df.columns:
        raise ValueError("Exploratory run blocked: missing final_qc_approval marker.")
    final_qc = df["final_qc_approval"].astype(str).str.strip().str.upper()
    if not final_qc.eq("NOT_GRANTED").all():
        raise ValueError("Exploratory run blocked: expected final_qc_approval=NOT_GRANTED.")
    print("WARNING: EXPLORATORY RESULTS ONLY; final QC approval has NOT been granted.")

class MRIOnlyDataset(Dataset):
    def __init__(self, df, mod):
        self.df = df.reset_index(drop=True)
        self.mod = mod
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.mod.load_mri(r.nifti_path)
        return {
            "x": torch.from_numpy(x),
            "label": torch.tensor(int(r.label), dtype=torch.long),
            "ptid": str(r.PTID),
        }

class ClinicalOnlyDataset(Dataset):
    def __init__(self, df, mean, std):
        self.df = df.reset_index(drop=True)
        self.mean = float(mean)
        self.std = max(float(std), 1e-6)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = (float(r.MMSE) - self.mean) / self.std
        return {
            "x": torch.tensor([x], dtype=torch.float32),
            "label": torch.tensor(int(r.label), dtype=torch.long),
            "ptid": str(r.PTID),
        }

class Baseline(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))

def evaluate(model, loader, device):
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            y = b["label"].to(device)
            logits = model(x)
            losses.append(float(F.cross_entropy(logits, y).cpu()))
            ys.extend(y.cpu().tolist())
            ps.extend(logits.argmax(1).cpu().tolist())

    cm = np.zeros((3, 3), dtype=int)
    for y, p in zip(ys, ps):
        cm[y, p] += 1

    f1s, recalls = [], []
    for c in range(3):
        tp = int(((np.array(ys) == c) & (np.array(ps) == c)).sum())
        fp = int(((np.array(ys) != c) & (np.array(ps) == c)).sum())
        fn = int(((np.array(ys) == c) & (np.array(ps) != c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)

    return {
        "n": len(ys),
        "accuracy": float(np.mean(np.array(ys) == np.array(ps))),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix_true_rows_pred_cols": cm.tolist(),
        "cross_entropy": float(np.mean(losses)),
    }

def run_one(name, df, mod, outroot, args, device):
    seed_all(args.seed)
    tr, va, te = (df[df.split == s].copy() for s in ("train", "validation", "test"))
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"{name}: one of the train/validation/test splits is empty.")

    if name == "MRI_only":
        encoder = mod.MRIEncoder(64)
        train_ds = MRIOnlyDataset(tr, mod)
        val_ds = MRIOnlyDataset(va, mod)
        test_ds = MRIOnlyDataset(te, mod)
        dim = 64
        standardizer = None
    else:
        mean = float(tr.MMSE.mean())
        std = float(tr.MMSE.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            raise ValueError("Invalid MMSE standard deviation in training split.")
        encoder = mod.ClinicalEncoder(16)
        train_ds = ClinicalOnlyDataset(tr, mean, std)
        val_ds = ClinicalOnlyDataset(va, mean, std)
        test_ds = ClinicalOnlyDataset(te, mean, std)
        dim = 16
        standardizer = {"mmse_mean_train": mean, "mmse_std_train": std}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Baseline(encoder, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = outroot / name
    out.mkdir(parents=True, exist_ok=True)
    best, best_ep, stale, hist = float("inf"), 0, 0, []
    checkpoint = out / "best_model.pt"

    print(f"\n=== {name} ===")
    print(f"Rows train/validation/test: {len(tr)}/{len(va)}/{len(te)}")
    for ep in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for b in train_loader:
            x, y = b["x"].to(device), b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite training loss at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        vm = evaluate(model, val_loader, device)
        hist.append({
            "epoch": ep,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_cross_entropy": vm["cross_entropy"],
            "validation_accuracy": vm["accuracy"],
            "validation_macro_f1": vm["macro_f1"],
            "validation_balanced_accuracy": vm["balanced_accuracy"],
        })
        print(json.dumps(hist[-1]))
        if vm["cross_entropy"] < best:
            best, best_ep, stale = vm["cross_entropy"], ep, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep, "seed": args.seed,
                "standardizer": standardizer,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {ep}; best epoch {best_ep}")
                break

    if not checkpoint.exists():
        raise RuntimeError(f"{name}: no best checkpoint was saved.")
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    summary = {
        "experiment": name,
        "input": "MRI only" if name == "MRI_only" else "MMSE only",
        "best_epoch": best_ep,
        "best_validation_cross_entropy": best,
        "test_metrics": test_metrics,
        "test_used_for_checkpoint_selection": False,
        "seed": args.seed,
        "rows_by_split": {s: int((df.split == s).sum()) for s in ("train", "validation", "test")},
        "participants_by_split": {
            s: int(df.loc[df.split == s, "PTID"].nunique())
            for s in ("train", "validation", "test")
        },
        "label_map": mod.LABEL_MAP,
        "limitations": ["Single run; not five-fold results", "Not a clinical validation study"],
    }
    if standardizer:
        summary["clinical_standardization_fit_on_train_only"] = True
        summary["clinical_standardizer"] = standardizer
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def export_combined(root):
    names = ["RGAF_SHMR", "MRI_only", "MMSE_only"]
    rows = []
    for name in names:
        p = root / name / "evaluation_summary.json"
        if not p.is_file():
            print(f"Not yet available: {p}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("test_metrics", {})
        rows.append({
            "experiment": d.get("experiment", name),
            "best_epoch": d.get("best_epoch"),
            "best_validation_cross_entropy": d.get("best_validation_cross_entropy"),
            "test_n": m.get("n"),
            "test_accuracy": m.get("accuracy"),
            "test_macro_f1": m.get("macro_f1"),
            "test_balanced_accuracy": m.get("balanced_accuracy"),
            "test_cross_entropy": m.get("cross_entropy"),
            "test_confusion_matrix_true_rows_pred_cols": json.dumps(
                m.get("confusion_matrix_true_rows_pred_cols")
            ),
            "test_used_for_checkpoint_selection": d.get("test_used_for_checkpoint_selection"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(root / "R2C3_combined_results.csv", index=False)
        (root / "R2C3_combined_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("Combined results exported:", root / "R2C3_combined_results.csv")
    else:
        print("No evaluation summaries available to combine yet.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outroot", default=str(DEFAULT_ROOT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.outroot)
    root.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        export_combined(root)
        return

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    mod = load_training_module()
    df = mod.read_manifest(str(manifest))
    check_eligibility(df)

    # Ensure all three experiments receive exactly the same participants/splits.
    if df.PTID.duplicated().any():
        raise ValueError("Duplicate PTIDs found; expected one row per participant.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("This runner trains only the two baselines; run RGAF+SHMR separately.")
    run_one("MRI_only", df, mod, root, args, device)
    run_one("MMSE_only", df, mod, root, args, device)
    export_combined(root)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
