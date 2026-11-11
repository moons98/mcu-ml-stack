"""SEU pre-training (paper Sec. 3.1): Adam, lr=1e-4, batch 32, 100 epochs, sparse CE."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "models/res_cnn_lstm"))
from model import build_res_cnn_lstm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/seu/processed/seu_random.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/seu_training/runs/seu_random")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    d = np.load(args.data)
    model = build_res_cnn_lstm(input_shape=d["x_train"].shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(args.out / "best.h5"), monitor="val_accuracy",
                                              save_best_only=True)
    hist = model.fit(d["x_train"], d["y_train"], validation_data=(d["x_val"], d["y_val"]),
                     epochs=args.epochs, batch_size=args.batch, callbacks=[ckpt], verbose=2)
    model.save(args.out / "last.h5")

    best = tf.keras.models.load_model(args.out / "best.h5")
    y_pred = best.predict(d["x_test"], verbose=0).argmax(1)
    y = d["y_test"]
    res = {
        "accuracy": float((y_pred == y).mean() * 100),
        "recall_macro": float(recall_score(y, y_pred, average="macro") * 100),
        "f1_macro": float(f1_score(y, y_pred, average="macro") * 100),
        "confusion": confusion_matrix(y, y_pred).tolist(),
        "params": int(best.count_params()),
        "h5_size_kb": round((args.out / "best.h5").stat().st_size / 1024, 1),
    }
    print(classification_report(y, y_pred, target_names=["health", "inner", "ball", "outer"], digits=4))
    print(json.dumps(res, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    (args.out / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}))


if __name__ == "__main__":
    main()
