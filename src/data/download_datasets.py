import csv
import json
import shutil
import gzip
import requests
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, out_path: Path, timeout=60) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return True
    except Exception as e:
        print(f"[WARN] Download failed: {url}\n  -> {e}")
        return False

def tsv_to_csv(tsv_path: Path, csv_path: Path):
    # Handle gzip-compressed TSV
    with gzip.open(tsv_path, "rt", encoding="utf-8", errors="ignore") as fin, \
         open(csv_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row)

def main():
    manifest = {"mathur": {}, "yada": {}}

    # -------------------------
    # 1) Mathur et al. dataset
    # Repo: aruneshmathur/dark-patterns  (NOT darkpatterns/dark-patterns)
    # File: data/final-dark-patterns/dark-patterns.csv
    # -------------------------
    mathur_url = (
        "https://raw.githubusercontent.com/aruneshmathur/dark-patterns/master/"
        "data/final-dark-patterns/dark-patterns.csv"
    )
    mathur_out = RAW_DIR / "mathur_dark_patterns.csv"

    if download_file(mathur_url, mathur_out):
        manifest["mathur"] = {"status": "downloaded", "path": str(mathur_out), "source": mathur_url}
        print(f"[OK] Mathur dataset downloaded -> {mathur_out}")
    else:
        manifest["mathur"] = {"status": "manual_required", "path": str(mathur_out), "source": mathur_url}
        print("\n[MANUAL REQUIRED] Mathur dataset download failed.")
        print("Open repo and download: aruneshmathur/dark-patterns")
        print("File path: data/final-dark-patterns/dark-patterns.csv")

    # -------------------------
    # 2) Yada et al. dataset
    # Repo: yamanalab/ec-darkpattern
    # File: dataset/dataset.tsv
    # We'll download TSV then convert to CSV
    # -------------------------
    yada_tsv_url = (
        "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/"
        "dataset/dataset.tsv"
    )
    yada_tsv_out = RAW_DIR / "yada_dark_patterns.tsv"
    yada_csv_out = RAW_DIR / "yada_dark_patterns.csv"

    if download_file(yada_tsv_url, yada_tsv_out):
        tsv_to_csv(yada_tsv_out, yada_csv_out)
        manifest["yada"] = {
            "status": "downloaded",
            "path_tsv": str(yada_tsv_out),
            "path_csv": str(yada_csv_out),
            "source": yada_tsv_url,
        }
        print(f"[OK] Yada dataset TSV downloaded -> {yada_tsv_out}")
        print(f"[OK] Converted to CSV -> {yada_csv_out}")
    else:
        manifest["yada"] = {"status": "manual_required", "path": str(yada_csv_out), "source": yada_tsv_url}
        print("\n[MANUAL REQUIRED] Yada dataset download failed.")
        print("Open repo and download: yamanalab/ec-darkpattern")
        print("File path: dataset/dataset.tsv (then convert to CSV)")

    # Save manifest
    out_manifest = RAW_DIR / "download_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Wrote manifest: {out_manifest}")

if __name__ == "__main__":
    main()