#!/usr/bin/env python3
import io, os, argparse, numpy as np, pandas as pd, requests, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

RAW_BASE = "https://raw.githubusercontent.com/subiksha03/dataset/main"
OUT_CSV = "dataset.csv"
METRICS_CSV = "metrics.csv"
REPORT_MD = "report.md"

CSV_CANDIDATES = ["api.csv", "ent.csv", "exe.csv", "strings2.csv"]
LABEL_NAME_CANDIDATES = ["label", "class", "target", "y"]

def _fetch(path: str) -> bytes:
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def fetch_to_csv() -> pd.DataFrame:
    try:
        excel_bytes = _fetch("final%20ds.xlsx")
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        sheet_name = None
        for s in xls.sheet_names:
            if str(s).strip().lower() == "dataset":
                sheet_name = s
                break
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)
        print(f"[INFO] Loaded Excel {sheet_name} with shape {df.shape}")
    except Exception as e:
        print(f"[INFO] Excel failed ({e}). Trying CSVs...")
        dfs = []
        for name in CSV_CANDIDATES:
            try:
                b = _fetch(name)
                part = pd.read_csv(io.BytesIO(b))
                dfs.append(part)
                print(f"[INFO] Loaded {name} {part.shape}")
            except Exception as ee:
                print(f"[INFO] Skip {name}: {ee}")
        if not dfs:
            raise RuntimeError("Could not load dataset.")
        df = pd.concat(dfs, axis=1)

    df.columns = [str(c).lower() for c in df.columns]
    label_col = None
    for c in df.columns:
        if c in LABEL_NAME_CANDIDATES:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("No label column found.")
    df = df.rename(columns={label_col: "label"})
    feat_cols = [c for c in df.columns if c != "label"]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV} shape={df.shape}")
    return df

def plot_cm(cm, classes, title, outpath):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run_rf(df: pd.DataFrame, runs: int = 5, test_size: float = 0.25):
    X = df.drop(columns=["label"])
    y = df["label"]
    rows = []
    with open(REPORT_MD, "w", encoding="utf-8") as report:
        report.write("# RandomForest Baseline Report\n\n")
        for run in range(1, runs + 1):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=run, stratify=y
            )
            rf = RandomForestClassifier(
                n_estimators=100 + 20 * run,
                max_depth=None if run % 2 else 12,
                min_samples_leaf=1 if run < 3 else 2,
                random_state=run, n_jobs=-1
            )
            rf.fit(X_tr, y_tr)
            y_pr = rf.predict(X_te)
            acc = accuracy_score(y_te, y_pr)
            pre = precision_score(y_te, y_pr, zero_division=0)
            rec = recall_score(y_te, y_pr, zero_division=0)
            f1  = f1_score(y_te, y_pr, zero_division=0)
            print(f"\n=== Run {run} ===")
            rep = classification_report(y_te, y_pr, digits=3)
            print(rep)
            cm = confusion_matrix(y_te, y_pr)
            cm_file = f"confusion_matrix_run{run}.png"
            plot_cm(cm, ["benign(0)", "ransom(1)"], f"RF CM (run {run})", cm_file)
            rows.append({
                "run": run,
                "accuracy": round(acc,4),
                "precision": round(pre,4),
                "recall": round(rec,4),
                "f1": round(f1,4)
            })
            report.write(f"## Run {run}\n```\n{rep}\n```\n![CM]({cm_file})\n\n")
    res = pd.DataFrame(rows)
    res.to_csv(METRICS_CSV, index=False)
    print("\nSummary:")
    print(res)
    print(f"[OK] wrote {METRICS_CSV}, {REPORT_MD}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()
    df = fetch_to_csv()
    run_rf(df, runs=args.runs, test_size=args.test_size)

if __name__ == "__main__":
    main()
