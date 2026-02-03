import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ---------------- CONFIG ----------------
DATA_PATH = Path("data/processed/train_binary.csv")
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
# ---------------------------------------


def encode_texts(tokenizer, texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )

    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_enc = encode_texts(tokenizer, X_train)
    val_enc = encode_texts(tokenizer, X_val)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (dict(train_enc), y_train)
    ).shuffle(1000).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (dict(val_enc), y_val)
    ).batch(BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss)

    print("[INFO] Training Transformer...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("[INFO] Evaluating...")
    logits = model.predict(val_ds).logits
    preds = np.argmax(logits, axis=1)

    f1 = f1_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)

    results = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "history": {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        }
    }

    out_json = OUT_DIR / "transformer_roberta_tf_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    model_dir = OUT_DIR / "roberta_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("[OK] Transformer training complete")
    print("[OK] Results saved to:", out_json)
    print("[OK] Model saved to:", model_dir)
    print("F1 score:", f1)


if __name__ == "__main__":
    main()