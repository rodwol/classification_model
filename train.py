import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from datetime import datetime

def load_data(x_path="data/processed/X.npy", y_path="processed/y.npy"):
    X = np.load(x_path)  # expect shape (N, H, W, C)
    y = np.load(y_path)
    # normalize (safety)
    X = X.astype("float32")
    if X.max() > 1.0:
        X = X / 255.0
    return X, y

def build_model(input_shape=(224,224,3), num_classes=2, weights=None):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg")
    x = base.output
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model

def main(args):
    X, y = load_data(args.x, args.y)

    # Optionally convert to binary (crackle vs normal) if requested
    if args.binary:
        # map labels >0 to 1
        y = (y != 0).astype(int)
        num_classes = 2
    else:
        num_classes = int(max(y) + 1)

    # train/val/test split
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    val_split = int(N * args.val_split)
    test_split = int(N * (args.val_split + args.test_split))

    X_train, y_train = X[test_split:], y[test_split:]
    X_val, y_val = X[val_split:test_split], y[val_split:test_split]
    X_test, y_test = X[:val_split], y[:val_split]

    # Data augmentation as tf.data pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    AUTOTUNE = tf.data.AUTOTUNE
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        # random crop / resize could be added
        return image, label

    train_ds = (train_ds.shuffle(1024)
                        .map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
                        .map(augment, num_parallel_calls=AUTOTUNE)
                        .batch(args.batch_size)
                        .prefetch(AUTOTUNE))

    val_ds = (val_ds.map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
                  .batch(args.batch_size)
                  .prefetch(AUTOTUNE))

    test_ds = (test_ds.map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
                   .batch(args.batch_size)
                   .prefetch(AUTOTUNE))

    model = build_model(input_shape=X.shape[1:], num_classes=num_classes, weights=args.weights)

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Directories & versioning
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = models_dir / f"run_{timestamp}"
    run_dir.mkdir()

    # Callbacks
    ckpt_path = run_dir / "ckpt.h5"
    cb_checkpoint = callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_loss")
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    csv_log = callbacks.CSVLogger(run_dir / "training_log.csv")

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=[cb_checkpoint, cb_early, cb_reduce, csv_log])

    # final evaluation
    eval_res = model.evaluate(test_ds)
    print("Test evaluation:", eval_res)

    # Save final model (saved_model + keras H5)
    model.save(run_dir / "model_saved")
    model.save(run_dir / "model.h5")
    # save metadata
    with open(run_dir / "meta.txt", "w") as f:
        f.write(f"num_classes={num_classes}\n")
        f.write(f"eval_res={eval_res}\n")
        f.write(f"binary={args.binary}\n")

    print("Model and artifacts saved to:", run_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--x", default="data/processed/X.npy")
    p.add_argument("--y", default="data/processed/y.npy")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--weights", default=None, help="imagenet or None") 
    p.add_argument("--binary", action="store_true", help="train binary crackle vs normal")
    args = p.parse_args()
    main(args)
