import os

# FIX 1: disable oneDNN/MKL (prevents memory object crash)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# FIX 2: limit CPU threads (prevents MKL memory explosion)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(x_path="data/processed/X.npy", y_path="data/processed/y.npy"):
    X = np.load(x_path).astype("float32")
    y = np.load(y_path)

    if X.max() > 1.0:
        X = X / 255.0

    print("Loaded X shape:", X.shape)   # ⭐ debug shape
    print("Loaded y shape:", y.shape)

    return X, y


def main(args):
    X, y = load_data(args.x, args.y)

    # FIX 3: load h5 without compilation issues
    model = tf.keras.models.load_model(args.model, compile=False)

    # FIX 4: safe small batch size to avoid MKL crash
    preds = model.predict(X, batch_size=8)

    y_pred = preds.argmax(axis=1)

    # metrics
    report = classification_report(y, y_pred, digits=4)
    print(report)

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(args.out_dir) / "classification_report.txt", [report], fmt="%s")
    np.save(Path(args.out_dir) / "confusion_matrix.npy", cm)

    # plot cm
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.savefig(Path(args.out_dir) / "confusion_matrix.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--x", default="data/processed/X.npy")
    p.add_argument("--y", default="data/processed/y.npy")
    p.add_argument("--out_dir", default="eval")
    args = p.parse_args()
    main(args)

