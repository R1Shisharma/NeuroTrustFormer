import argparse, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TRAIN = str(SCRIPT_DIR / "train_adni_rgaf_shmr.py")
DEFAULT_BASE = str(SCRIPT_DIR / "r2c3_baselines_and_export_EXPLORATORY.py")


LABELS = {"CN": 0, "MCI": 1, "AD": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exploratory ADNI five-fold experiments."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the five fold manifests."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where experiment outputs will be saved."
    )
    parser.add_argument(
        "--training-script",
        type=Path,
        default=SCRIPT_DIR / "train_adni_rgaf_shmr.py"
    )
    parser.add_argument(
        "--baseline-script",
        type=Path,
        default=SCRIPT_DIR / "r2c3_baselines_and_export_EXPLORATORY.py"
    )
    return parser.parse_args()


def stratified_folds(df, k, seed):
    rng = np.random.default_rng(seed); bins = [[] for _ in range(k)]
    y = df.diagnosis_label.map(LABELS).to_numpy()
    for lab in sorted(set(y)):
        ids = np.flatnonzero(y == lab); rng.shuffle(ids); start = int(lab) % k
        for j, ix in enumerate(ids): bins[(start+j) % k].append(int(ix))
    return [np.array(sorted(b), dtype=int) for b in bins]

def train_val(dev, frac, seed):
    rng = np.random.default_rng(seed); tr, va = [], []
    for lab in sorted(dev.diagnosis_label.unique()):
        ids = dev.index[dev.diagnosis_label == lab].to_numpy(copy=True); rng.shuffle(ids)
        if len(ids) < 2: raise ValueError(f"Too few {lab} rows for validation")
        n = min(len(ids)-1, max(1, int(round(len(ids)*frac))))
        va += ids[:n].tolist(); tr += ids[n:].tolist()
    return sorted(tr), sorted(va)

def run(cmd):
    print("\nRUNNING:", subprocess.list2cmdline([str(x) for x in cmd]), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

def main():
    ap = argparse.ArgumentParser(description="Exploratory participant-level 5-fold R2C3 orchestration")
    ap.add_argument("--manifest", required=True); ap.add_argument("--outroot", required=True)
    ap.add_argument("--train-script", default=DEFAULT_TRAIN); ap.add_argument("--baseline-script", default=DEFAULT_BASE)
    ap.add_argument("--python-launcher", default="py"); ap.add_argument("--python-version", default="-3.12")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--validation-fraction", type=float, default=.15); a = ap.parse_args()
    if not .05 <= a.validation_fraction <= .30: raise ValueError("validation-fraction must be 0.05–0.30")
    src = Path(a.manifest)
    if not src.is_file(): raise FileNotFoundError(src)
    d = pd.read_csv(src).copy()
    required = ["PTID","diagnosis_label","MMSE","nifti_path","approved_for_training","exploratory_only","final_qc_approval"]
    missing = [c for c in required if c not in d.columns]
    if missing: raise ValueError(f"Manifest missing required columns: {missing}")
    if d.PTID.astype(str).str.strip().duplicated().any(): raise ValueError("Expected one row per participant")
    if not d.exploratory_only.astype(str).str.lower().isin(["true","1","yes"]).all(): raise ValueError("Every row must be exploratory_only=True")
    if not d.final_qc_approval.astype(str).str.upper().eq("NOT_GRANTED").all(): raise ValueError("Expected final_qc_approval=NOT_GRANTED; do not bypass QC")
    if not d.approved_for_training.astype(str).str.lower().isin(["true","1","yes","approved"]).all(): raise ValueError("Existing runner requires approved_for_training=True; do not change QC flags to bypass")
    d["PTID"] = d.PTID.astype(str).str.strip(); d["diagnosis_label"] = d.diagnosis_label.astype(str).str.strip().str.upper()
    if not d.diagnosis_label.isin(LABELS).all(): raise ValueError("Unexpected diagnosis label")
    cnt = d.diagnosis_label.value_counts()
    if len(cnt) != 3 or cnt.min() < 5: raise ValueError(f"Cannot stratify five folds: {cnt.to_dict()}")
    root = Path(a.outroot); root.mkdir(parents=True, exist_ok=True)
    folds = stratified_folds(d, 5, a.seed)
    assignment = pd.DataFrame({"PTID": d.PTID, "diagnosis_label": d.diagnosis_label, "outer_test_fold": 0})
    for k, ids in enumerate(folds, 1): assignment.loc[ids, "outer_test_fold"] = k
    assignment.to_csv(root/"participant_fold_assignments.csv", index=False)
    rows = []
    for k, test_ids in enumerate(folds, 1):
        dev_ids = sorted(set(range(len(d))) - set(test_ids)); dev = d.iloc[dev_ids].copy()
        _, val_ids = train_val(dev, a.validation_fraction, a.seed+k*100)
        fold_df = d.copy(); fold_df["split"] = "train"
        fold_df.loc[val_ids, "split"] = "validation"; fold_df.loc[test_ids, "split"] = "test"
        sets = [set(fold_df.loc[fold_df.split==s, "PTID"]) for s in ("train","validation","test")]
        if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise RuntimeError(f"Participant overlap in fold {k}")
        fd = root/f"fold_{k}"; fd.mkdir(parents=True, exist_ok=True); mf = fd/"fold_manifest.csv"; fold_df.to_csv(mf,index=False)
        (fd/"split_summary.json").write_text(json.dumps({"fold":k,"train_n":len(sets[0]),"validation_n":len(sets[1]),"test_n":len(sets[2]),"participant_overlap":0,"qc_status":"EXPLORATORY_ONLY; final QC approval NOT_GRANTED"},indent=2),encoding="utf-8")
        print(f"\n========== FOLD {k}/5 ==========",flush=True)
        run([a.python_launcher,a.python_version,a.train_script,"--manifest",mf,"--outdir",fd/"RGAF_SHMR","--epochs",a.epochs,"--batch-size",a.batch_size,"--patience",a.patience,"--seed",a.seed+k*1000])
        run([a.python_launcher,a.python_version,a.baseline_script,"--manifest",mf,"--outroot",fd/"BASELINES","--epochs",a.epochs,"--batch-size",a.batch_size,"--patience",a.patience,"--seed",a.seed+k*1000])
        for name, p in [("RGAF_SHMR",fd/"RGAF_SHMR"/"evaluation_summary.json"),("MRI_only",fd/"BASELINES"/"MRI_only"/"evaluation_summary.json"),("MMSE_only",fd/"BASELINES"/"MMSE_only"/"evaluation_summary.json")]:
            if not p.is_file(): raise FileNotFoundError(f"Expected summary missing: {p}")
            obj=json.loads(p.read_text(encoding="utf-8")); m=obj.get("test_metrics",{})
            rows.append({"model":name,"fold":k,"n":m.get("n"),"accuracy":m.get("accuracy"),"macro_f1":m.get("macro_f1"),"balanced_accuracy":m.get("balanced_accuracy"),"cross_entropy":m.get("cross_entropy"),"best_epoch":obj.get("best_epoch"),"test_used_for_checkpoint_selection":False,"confusion_matrix":json.dumps(m.get("confusion_matrix_true_rows_pred_cols"))})
    result=pd.DataFrame(rows).sort_values(["model","fold"]); result.to_csv(root/"fivefold_metrics_by_model.csv",index=False)
    (root/"fivefold_metrics_by_model.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    agg=[]
    for name,g in result.groupby("model"):
        r={"model":name,"folds_completed":int(g.fold.nunique())}
        for m in ["accuracy","macro_f1","balanced_accuracy","cross_entropy"]:
            r[m+"_mean"]=float(g[m].mean()); r[m+"_sd_sample"]=float(g[m].std(ddof=1))
        agg.append(r)
    pd.DataFrame(agg).sort_values("model").to_csv(root/"fivefold_mean_sd_summary.csv",index=False)
    (root/"fivefold_mean_sd_summary.json").write_text(json.dumps(agg,indent=2),encoding="utf-8")
    print("\nEXPLORATORY FIVE-FOLD SUMMARY\n",pd.DataFrame(agg).to_string(index=False))
    print("\nQC flags unchanged. Results remain exploratory.")

if __name__ == "__main__": main()
