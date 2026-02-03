import json
from pathlib import Path

# Paths to result files (already produced by previous steps)
BASELINE = Path("experiments/baseline_tfidf_lr_results.json")
TRANSFORMER = Path("experiments/transformer_roberta_tf_results.json")

def main():
    if not BASELINE.exists():
        raise FileNotFoundError(
            "Baseline results not found. Run baseline_ml.py first."
        )
    if not TRANSFORMER.exists():
        raise FileNotFoundError(
            "Transformer results not found. Run transformer.py first."
        )

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    tr = json.loads(TRANSFORMER.read_text(encoding="utf-8"))

    print("\n================ MODEL COMPARISON (STEP 5) ================\n")

    print(f"Baseline Model: {base['model']}")
    print(f"  F1-score: {base['f1']:.4f}\n")

    print(f"Transformer Model: {tr['model']}")
    print(f"  Precision: {tr['precision']:.4f}")
    print(f"  Recall:    {tr['recall']:.4f}")
    print(f"  F1-score:  {tr['f1']:.4f}\n")

    # Save paper-ready comparison
    comparison = {
        "baseline": {
            "model": base["model"],
            "f1": base["f1"]
        },
        "transformer": {
            "model": tr["model"],
            "precision": tr["precision"],
            "recall": tr["recall"],
            "f1": tr["f1"]
        }
    }

    out_path = Path("experiments/model_comparison.json")
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("[OK] Comparison saved to:", out_path)
    print("\n===========================================================\n")

if __name__ == "__main__":
    main()