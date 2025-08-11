# SSE-TSR

SSE-TSR classifies proteins using TSR-derived keys augmented with secondary structure information. The pipeline:

1. fetches PDBs,
2. generates per-chain `.3Dkeys…` files (with SS typing),
3. extracts the set of **unique 3D keys**,
4. builds **sparse matrices** and prepares the `train/validation/test` dataset,
5. **trains/evaluates** a CNN classifier (binary or multiclass), printing metrics and saving plots.

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)

  * [Step 1: Retrieve PDB files](#step-1-retrieve-pdb-files)
  * [Step 2: Generate 3D keys + SS types](#step-2-generate-3d-keys--ss-types)
  * [Step 3: Extract unique keys](#step-3-extract-unique-keys)
  * [Step 4: Build sparse matrices + dataset split](#step-4-build-sparse-matrices--dataset-split)
  * [Step 5: Train & evaluate classifier](#step-5-train--evaluate-classifier)
* [Notes & Tips](#notes--tips)
* [Contributing](#contributing)

## Overview

**Inputs**

* `sample_details.csv` (or `sample_details_*.csv`): must contain at least `protein`, `chain`, and a class label column (default: `group1`).
* PDB files (downloaded in Step 1).

**Outputs**

* `.3Dkeys_theta…` per-chain files (Step 2)
* `unique_3D_keys.txt` (Step 3)
* `Transformed_sparse_matrices/*.npz` (Step 4)
* `processed_dataset/{train,validation,test}/class*/…` (Step 4)
* `secondary_structure_batch_model.h5` and `results_summary.png` (Step 5)

## Prerequisites

* Python 3.9+
* Packages:

  * `pandas`, `numpy`, `scikit-learn`, `scipy`
  * `tensorflow` / `keras`
  * `matplotlib`, `seaborn`
  * `joblib`
* Internet access for PDB download (Step 1)

Install everything via:

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone https://github.com/pooryakhajouie/SSE-TSR.git
cd SSE-TSR
pip install -r requirements.txt
```

Make sure your `sample_details.csv` (with `protein,chain,group1` columns) is in the project—or pass its path via CLI where applicable.

## Usage

### Step 1: Retrieve PDB files

Downloads PDBs for the `protein` codes in your CSV and writes a cleaned CSV excluding failed downloads.

```bash
# show help
python src/1_pdb_retrieve.py -h

# typical run
python src/1_pdb_retrieve.py sample_details.csv Dataset/ \
  -o sample_details_cleaned.csv -j 16 --overwrite
```

**Inputs**

* `sample_details.csv` with column `protein`.

**Outputs**

* PDBs in `Dataset/`
* `sample_details_cleaned.csv`

---

### Step 2: Generate 3D keys + SS types

Generates `.3Dkeys_theta30_maxdist35` per protein/chain and assigns one of 18 SS types per key.

```bash
python src/2_keyTransformation1D_3D_HelixSheetNone.py \
  --sample-csv sample_details_cleaned.csv \
  --pdb-dir Dataset/ \
  --out-dir Triplet_type/ \
  --amino-lex aminoAcidCode_lexicographic_new.txt \
  --amino-codes amino_codes.txt
```

> Use the flags your script supports; the example shows typical args.

**Outputs**

* `Triplet_type/{PROT}_{CHAIN}.3Dkeys_theta30_maxdist35`

---

### Step 3: Extract unique keys

Scans the `.3Dkeys…` files and writes a sorted set of unique key IDs.

```bash
# show help
python src/3_extract_unique_keys.py -h

# typical run
python src/3_extract_unique_keys.py sample_details_cleaned.csv Triplet_type/ \
  -o unique_3D_keys.txt -j 16
```

**Outputs**

* `unique_3D_keys.txt`

---

### Step 4: Build sparse matrices + dataset split

Builds a `(types x keys)` sparse matrix per protein/chain and splits into `train/validation/test` with `class1..classN` folders.

```bash
# show help
python src/4_generate_sparse_matrices_and_dataset.py -h

# typical run
python src/4_generate_sparse_matrices_and_dataset.py \
  sample_details_cleaned.csv \
  unique_3D_keys.txt \
  Triplet_type/ \
  Transformed_sparse_matrices/ \
  processed_dataset/ \
  -j 16 --test-size 0.10 --val-size 0.10
```

**Key options**

* `--class-col group1` (default) – change if your label column name differs
* `--filename-template "{protein}_{chain}.3Dkeys_theta30_maxdist35"` (default)

**Outputs**

* `Transformed_sparse_matrices/{PROT}_{CHAIN}_transformed_sparse_matrix.npz`
* `processed_dataset/{train,validation,test}/class*/…`

---

### Step 5: Train & evaluate classifier

Trains a Conv1D model on the sparse matrices and evaluates on the test split. Supports **binary and multiclass**. Prints the **confusion matrix** (counts + row-normalized) and saves a summary plot with accuracy/loss, confusion heatmap, and ROC curves.

```bash
# show help
python src/5_train_classifier.py -h

# auto-detect classes and input shape
python src/5_train_classifier.py processed_dataset/ \
  --sample-details sample_details_cleaned.csv \
  --epochs 50 --batch-size 300 --lr 1e-3 \
  --model-out secondary_structure_batch_model.h5 \
  --plot-out results_summary.png
```

**Notes**

* If you know the number of classes: `--num-classes 2` (binary) or e.g. `--num-classes 8`.
* **ROC/AUC**: binary → one ROC curve using positive class; multiclass → one-vs-rest ROC per class (skips classes absent in test set).
* Optionally run a sanity-check to ensure CSV labels match folder labels (see script comments).

## Notes & Tips

* **CSV columns**: At minimum, `protein`, `chain`, and a label column (default `group1`). If you change the label column name, pass it via `--class-col` in Step 4; Step 5 can optionally use `--sample-details` for checks.
* **Class folders**: Step 4 creates `class1..classN` based on sorted unique labels, mapping `label → index (0-based) → class{index+1}`.
* **Reproducibility**: pass `--random-state` in Step 4 and `--seed` in Step 5.
* **Performance**: use `-j` to increase parallelism in Steps 1 and 4.

## Contributing

PRs are welcome. Please open an issue first to discuss substantial changes.

