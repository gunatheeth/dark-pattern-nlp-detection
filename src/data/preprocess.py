import re
import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


# -------------------------
# Load Mathur dataset
# -------------------------
def load_mathur() -> pd.DataFrame:
    path = RAW_DIR / "mathur_dark_patterns.csv"
    if not path.exists():
        print("[WARN] Mathur dataset not found.")
        return pd.DataFrame(columns=["text", "label", "source", "pattern_type"])

    # read gzip csv
    df = pd.read_csv(path, compression="gzip")

    # ---- SMART TEXT COLUMN DETECTION ----
    # Priority: known likely names first
    candidate_cols = [
        "text", "description", "snippet", "content",
        "pattern_text", "element_text", "ui_text", "prompt",
        "example", "quote", "message"
    ]

    text_col = None
    for c in candidate_cols:
        if c in df.columns:
            text_col = c
            break

    # If still not found, pick the longest average string column automatically
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError(f"No object/text-like columns found. Columns: {list(df.columns)}")

        # choose column with max average length
        lengths = {}
        for c in obj_cols:
            lengths[c] = df[c].astype(str).str.len().mean()
        text_col = max(lengths, key=lengths.get)

        print(f"[INFO] Auto-selected Mathur text column: {text_col}")

    # Pattern column if available
    pattern_col = None
    for c in ["pattern_type", "pattern", "category", "dark_pattern_type"]:
        if c in df.columns:
            pattern_col = c
            break

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).map(clean_text)
    out["label"] = 1
    out["source"] = "mathur"
    out["pattern_type"] = df[pattern_col].astype(str) if pattern_col else ""

    return out


# -------------------------
# Load Yada dataset (TSV)
# -------------------------
def load_yada() -> pd.DataFrame:
    """
    Robust loader:
    - supports TSV or CSV
    - supports gzipped or plain
    - auto-detects delimiter if needed
    """

    # Prefer TSV first, else CSV
    tsv_path = RAW_DIR / "yada_dark_patterns.tsv"
    csv_path = RAW_DIR / "yada_dark_patterns.csv"

    if tsv_path.exists():
        path = tsv_path
        sep = "\t"
    elif csv_path.exists():
        path = csv_path
        sep = ","
    else:
        raise FileNotFoundError("No Yada dataset found in data/raw/")

    # Try reading with utf-8 first, then gzip if needed
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        # file is probably gzipped
        df = pd.read_csv(path, sep=sep, compression="gzip", encoding="utf-8", engine="python")

    # Detect columns
    text_col = None
    for c in ["text", "sentence", "ui_text", "content"]:
        if c in df.columns:
            text_col = c
            break

    label_col = None
    for c in ["label", "y", "target", "is_darkpattern"]:
        if c in df.columns:
            label_col = c
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Yada columns not recognized. Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).map(clean_text)
    out["label"] = df[label_col].astype(int)
    out["source"] = "yada"

    return out


# -------------------------
# Main preprocessing
# -------------------------
def main():
    print("[INFO] Loading datasets...")

    mathur = load_mathur()
    yada = load_yada()

    print(f"[INFO] Mathur rows (positive only): {len(mathur)}")
    print(f"[INFO] Yada rows (pos+neg): {len(yada)}")

    # Use Yada for training (has negatives)
    train_df = yada.copy()

    # Clean
    train_df = train_df[train_df["text"].str.len() > 0]
    train_df = train_df.drop_duplicates(subset=["text", "label"])

    # Save outputs
    train_out = OUT_DIR / "train_binary.csv"
    train_df.to_csv(train_out, index=False, encoding="utf-8")

    mathur_out = OUT_DIR / "mathur_positive_only.csv"
    mathur.to_csv(mathur_out, index=False, encoding="utf-8")

    print("[OK] Saved:")
    print(" -", train_out)
    print(" -", mathur_out)

    print("\n[STATS] Label distribution:")
    print(train_df["label"].value_counts())

    print("\n[SAMPLE]")
    print(train_df.sample(5)[["text", "label"]])


if __name__ == "__main__":
    main()