#!/usr/bin/env python3
import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CNN on sparse matrices stored under processed_dataset/, "
                    "evaluate, and plot metrics (handles binary and multiclass ROC)."
    )
    p.add_argument("npz_dir",
                   help="Directory with splits: processed_dataset/{train,validation,test}/class*/.")
    p.add_argument("--sample-details", default="sample_details.csv",
                   help="Path to sample-details CSV (for reference; not required for training).")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Number of classes. If omitted, inferred from train/ subfolders named class1..classN.")
    p.add_argument("--batch-size", type=int, default=300,
                   help="Batch size (default: 300).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Max epochs (default: 50).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience on val_loss (default: 10).")
    p.add_argument("--model-out", default="secondary_structure_batch_model.h5",
                   help="Where to save best model (default: secondary_structure_batch_model.h5).")
    p.add_argument("--plot-out", default="results_summary.png",
                   help="Where to save the summary plot (default: results_summary.png).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    return p.parse_args()


# ----------------------------
# Data helpers
# ----------------------------
def list_split_files(npz_dir, split):
    """Return (filepaths, labels) for a split. Labels are 0..C-1 based on classX folder names."""
    subdir = os.path.join(npz_dir, split)
    if not os.path.isdir(subdir):
        raise SystemExit(f"Missing split folder: {subdir}")

    files, labels = [], []
    class_folders = [d for d in sorted(os.listdir(subdir)) if d.startswith("class")]
    for class_folder in class_folders:
        label = int(class_folder.split('class')[-1]) - 1
        folder_path = os.path.join(subdir, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for fn in sorted(os.listdir(folder_path)):
            if fn.endswith(".npz"):
                files.append(os.path.join(folder_path, fn))
                labels.append(label)
    return np.array(files), np.array(labels, dtype=np.int32)


def infer_input_shape(npz_dir):
    """Load one file from train/ (fallback to validation/ then test/) to infer (rows, cols)."""
    for split in ["train", "validation", "test"]:
        files, _ = list_split_files(npz_dir, split)
        if len(files):
            shape = load_npz(files[0]).shape  # (rows, cols)
            return shape
    raise SystemExit("Could not infer input shape; no .npz files found in any split.")


def infer_num_classes(npz_dir):
    """Count the number of class folders in train/ (class1..classN)."""
    train_dir = os.path.join(npz_dir, "train")
    class_folders = [d for d in os.listdir(train_dir) if d.startswith("class")]
    if not class_folders:
        raise SystemExit(f"No class folders found under {train_dir}")
    idxs = [int(d.split('class')[-1]) for d in class_folders]
    return max(idxs)


def data_generator(npz_dir, batch_size, mode='train', shuffle=True):
    """
    Yields (X_batch, y_batch):
      X_batch: (B, rows, cols) float32
      y_batch: (B,) int labels
    """
    assert mode in ('train', 'validation', 'test')
    files, labels = list_split_files(npz_dir, mode)
    n = len(files)
    if n == 0:
        raise SystemExit(f"No samples found in split '{mode}' under {npz_dir}")

    while True:
        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idxs = idxs[start:start + batch_size]
            Xb = [load_npz(files[i]).toarray().astype('float32') for i in batch_idxs]
            Xb = np.stack(Xb, axis=0)
            yb = labels[batch_idxs]
            yield Xb, yb


def count_samples_in_split(npz_dir, split):
    files, _ = list_split_files(npz_dir, split)
    return len(files)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape, num_classes, lr):
    """
    Builds a Conv1D model similar to your original, but with dynamic input shape and output units.
    input_shape: (rows, cols) -> interpreted as (timesteps=rows, features=cols)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.1)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.1))
    ])

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Optional: load sample-details just so it's flexible per your request
    if args.sample_details and os.path.exists(args.sample_details):
        try:
            labels_df = pd.read_csv(args.sample_details)
            # Not strictly used for training, but kept for reference/debugging
            if not {'protein', 'group1'}.issubset(labels_df.columns):
                print(f"[INFO] Loaded {args.sample_details}, but missing typical columns; that's okay.")
        except Exception as e:
            print(f"[WARN] Could not read sample-details CSV ({args.sample_details}): {e}")

    # Determine classes & input shape
    num_classes = args.num_classes or infer_num_classes(args.npz_dir)
    input_rows, input_cols = infer_input_shape(args.npz_dir)
    print(f"[INFO] Detected input shape: ({input_rows}, {input_cols}); num_classes = {num_classes}")

    # Steps per epoch
    train_count = count_samples_in_split(args.npz_dir, 'train')
    val_count   = count_samples_in_split(args.npz_dir, 'validation')
    test_count  = count_samples_in_split(args.npz_dir, 'test')

    train_steps = math.ceil(train_count / args.batch_size)
    val_steps   = math.ceil(val_count   / args.batch_size)
    test_steps  = math.ceil(test_count  / args.batch_size)

    # Class weights from train distribution
    _, train_labels = list_split_files(args.npz_dir, 'train')
    classes = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Computed class_weight:")
    for c, w in class_weight.items():
        print(f"  class {c}: {w:.3f}")

    # Build & train
    model = build_model((input_rows, input_cols), num_classes, args.lr)
    print(model.summary())

    es  = EarlyStopping(monitor='val_loss', mode='min', patience=args.patience, restore_best_weights=True)
    rl  = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=1e-4)
    ck  = ModelCheckpoint(args.model_out, monitor='val_loss', mode='min', save_best_only=True)
    
    sanity_check_labels(args.npz_dir, args.sample_details, class_col='group1', strict=True)

    history = model.fit(
        data_generator(args.npz_dir, args.batch_size, mode='train', shuffle=True),
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_data=data_generator(args.npz_dir, args.batch_size, mode='validation', shuffle=False),
        validation_steps=val_steps,
        class_weight=class_weight,
        callbacks=[es, rl, ck],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(
        data_generator(args.npz_dir, args.batch_size, mode='test', shuffle=False),
        steps=test_steps,
        verbose=1
    )
    print(f"[RESULT] Test accuracy: {test_acc:.4f}")

    # Collect predictions on the test set
    y_true_list, y_pred_list, y_prob_list = [], [], []
    test_gen = data_generator(args.npz_dir, args.batch_size, mode='test', shuffle=False)
    for _ in range(test_steps):
        Xb, yb = next(test_gen)
        pb = model.predict(Xb, verbose=0)             # (B, C)
        y_true_list.append(yb)
        y_pred_list.append(np.argmax(pb, axis=1))
        y_prob_list.append(pb)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_prob = np.vstack(y_prob_list)

    # Confusion matrix
    conf_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion matrix (counts):")
    print(conf_raw)

    # Row-normalized (each row sums to 1.0)
    conf_norm = conf_raw.astype(np.float32)
    conf_norm = conf_norm / np.clip(conf_norm.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

    # Optional: prettier printing
    np.set_printoptions(precision=3, suppress=True)
    print("\nConfusion matrix (row-normalized):")
    print(conf_norm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(num_classes)), digits=4))

    # --------- Plotting ---------
    plt.figure(figsize=(10, 8))

    # 1) Train/Val Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 2) Train/Val Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 3) Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(conf_norm, fmt=".2f", cmap='Blues',
                xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)))
    plt.title('Confusion Matrix (row-normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 4) ROC / AUC
    plt.subplot(2, 2, 4)

    if num_classes == 2:
        # For binary: use the positive class score (assume class "1" is positive)
        y_score = y_prob[:, 1]  # softmax column 1
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_val:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.title('ROC Curve (Binary)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
    else:
        # Multiclass: One-vs-Rest curves
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        cmap = plt.cm.get_cmap('tab10', num_classes)
        any_plotted = False
        for i in range(num_classes):
            # Skip if no positives for this class in the test set (ROC undefined)
            if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == y_true_bin.shape[0]:
                continue
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = auc(fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, color=cmap(i), lw=2, label=f"Class {i} (AUC={auc_i:.3f})")
            any_plotted = True
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.title(f'One-vs-Rest ROC Curves ({num_classes} classes)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if any_plotted:
            plt.legend(loc='lower right', fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=300)
    print(f"[SAVED] Plot -> {args.plot_out}")
    print(f"[SAVED] Model -> {args.model_out}")


if __name__ == "__main__":
    main()

