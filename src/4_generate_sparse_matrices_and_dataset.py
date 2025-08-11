#!/usr/bin/env python3
import os
import csv
import argparse
import shutil
import multiprocessing
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split


# --- Default order of triplet-type columns (matches your original list) ---
TYPE_COLUMNS_DEFAULT = [
    '1_3a1', '2_3a2', '3_3a3', '4_2a11c', '5_2a21c', '6_1a2c ',
    '7_3b1', '8_3b2', '9_3b3', '10_2b11c', '11_2b21c', '12_1b2c',
    '13_2a21b', '14_2a11b', '15_2b21a', '16_2b11a', '17_3c', '18_1a1b1c'
]


def parse_args():
    p = argparse.ArgumentParser(
        description="1) Build sparse matrices from per-chain .3Dkeys files; "
                    "2) Split into train/test/validation folders."
    )
    # Required positionals (explicit beats accidental defaults)
    p.add_argument("sample_details",
                   help="CSV with at least columns: protein, chain, and a class column (default name: group1).")
    p.add_argument("keys_file",
                   help="Path to text file containing one integer 3D key per line (ordering preserved).")
    p.add_argument("key_file_dir",
                   help="Directory containing per-chain .3Dkeys files (e.g., Triplet_type/).")
    p.add_argument("sparse_out_dir",
                   help="Output directory for per-protein sparse matrices (.npz).")
    p.add_argument("dataset_out_dir",
                   help="Output directory for processed_dataset/ (train/test/validation).")

    # Practical tweaks
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                   help="Worker processes for the transform step (default: CPU count).")
    p.add_argument("--filename-template",
                   default="{protein}_{chain}.3Dkeys_theta30_maxdist35",
                   help="Relative filename template under triplet_dir (default: {protein}_{chain}.3Dkeys_theta30_maxdist35).")
    p.add_argument("--class-col", default="group1",
                   help="Class label column in mapping CSV (default: group1).")
    p.add_argument("--protein-col", default="protein",
                   help="Protein code column in mapping CSV (default: protein).")
    p.add_argument("--chain-col", default="chain",
                   help="Chain column in mapping CSV (default: chain).")
    p.add_argument("--test-size", type=float, default=0.10,
                   help="Fraction for test split (default: 0.10).")
    p.add_argument("--val-size", type=float, default=0.10,
                   help="Fraction for validation split (default: 0.10).")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed for splits (default: 42).")
    return p.parse_args()


def load_unique_keys(keys_file):
    unique_keys = []
    with open(keys_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            unique_keys.append(int(s))
    return unique_keys


def read_3dkeys_file(file_path):
    """
    Reads a .3Dkeys_* file and returns:
      freq_dict: { key(int): { type_name(str): float_frequency } }
      header_types: [type_name, ...] as found in the file header (columns 2..end)
    """
    freq_dict = {}
    with open(file_path, "r") as f:
        header = next(f).strip().split("\t")
        header_types = header[1:]
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            key = int(cols[0])
            type_freqs = {}
            for i, freq in enumerate(cols[1:]):
                type_name = header_types[i]
                try:
                    type_freqs[type_name] = float(freq)
                except ValueError:
                    type_freqs[type_name] = 0.0
            freq_dict[key] = type_freqs
    return freq_dict, header_types


def process_one(protein_name, chain_name, triplet_dir, filename_template, unique_keys, type_columns, sparse_out_dir):
    """
    Build a (types x keys) matrix for one (protein, chain) and save CSR .npz.
    """
    rel = filename_template.format(protein=protein_name, chain=chain_name)
    src = os.path.join(triplet_dir, rel)
    if not os.path.exists(src):
        print(f"[WARN] .3Dkeys file not found, skipping: {src}")
        return False

    freq_dict, _ = read_3dkeys_file(src)

    # Build rows in the fixed type order; each row is over all unique keys
    rows = []
    for tname in type_columns:
        row = []
        for k in unique_keys:
            if k in freq_dict:
                row.append(freq_dict[k].get(tname, 0.0))  # robust to small header mismatches
            else:
                row.append(0.0)
        rows.append(row)

    # Save sparse
    mat = csr_matrix(rows)
    os.makedirs(sparse_out_dir, exist_ok=True)
    out_name = f"{protein_name}_{chain_name}_sparse_matrix.npz"
    out_path = os.path.join(sparse_out_dir, out_name)
    save_npz(out_path, mat)
    return True


def make_chain_dict(df, protein_col, chain_col):
    """
    Mimics original behavior: one chain per protein (last occurrence wins).
    """
    chain_dict = {}
    for _, r in df[[protein_col, chain_col]].iterrows():
        chain_dict[str(r[protein_col])] = str(r[chain_col])
    return chain_dict


def ensure_class_folders(base_dir, class_index_to_name):
    for subset in ["train", "test", "validation"]:
        subset_dir = os.path.join(base_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for cname in class_index_to_name.values():
            os.makedirs(os.path.join(subset_dir, cname), exist_ok=True)


def move_matrices(split_df, subset_name, protein_col, chain_col, class_col,
                  sparse_out_dir, dataset_out_dir, label_to_index):
    """
    Copy saved .npz for each row into processed_dataset/<subset>/classX/
    """
    subset_dir = os.path.join(dataset_out_dir, subset_name)
    for _, row in split_df.iterrows():
        protein = str(row[protein_col])
        chain = str(row[chain_col])
        label = row[class_col]

        # Folder name is class{index+1}
        idx = label_to_index[label]
        class_folder = f"class{idx + 1}"

        fname = f"{protein}_{chain}_sparse_matrix.npz"
        src = os.path.join(sparse_out_dir, fname)
        dst = os.path.join(subset_dir, class_folder, fname)

        if not os.path.exists(src):
            print(f"[WARN] Matrix missing for {protein}_{chain}; looked for {src}. Skipping.")
            continue

        shutil.copy(src, dst)


def main():
    args = parse_args()

    # --- Validate split sizes ---
    tv = args.test_size + args.val_size
    if not (0 < args.test_size < 1 and 0 < args.val_size < 1 and tv < 1):
        raise SystemExit("test-size and val-size must be in (0,1) and their sum must be < 1.")

    # --- Load mapping CSV ---
    df = pd.read_csv(args.sample_details)
    for col in [args.protein_col, args.chain_col, args.class_col]:
        if col not in df.columns:
            raise SystemExit(f"Required column '{col}' not found in {args.mapping_csv}. "
                             f"Available: {list(df.columns)}")

    # --- Prepare protein/chain lists (one chain per protein, last occurrence wins) ---
    chain_dict = make_chain_dict(df, args.protein_col, args.chain_col)
    protein_names = sorted(chain_dict.keys())

    # --- Load keys & set type order ---
    unique_keys = load_unique_keys(args.keys_file)
    type_columns = TYPE_COLUMNS_DEFAULT  # maintain your original order

    # --- Build sparse matrices in parallel ---
    print(f"[INFO] Building sparse matrices for {len(protein_names)} proteins using {args.jobs} workers …")
    work = [
        (p, chain_dict[p], args.key_file_dir, args.filename_template,
         unique_keys, type_columns, args.sparse_out_dir)
        for p in protein_names
    ]
    jobs = max(1, int(args.jobs or 1))
    with multiprocessing.Pool(processes=jobs) as pool:
        results = pool.starmap(process_one, work)

    built = sum(1 for x in results if x)
    print(f"[INFO] Sparse matrices created: {built} / {len(protein_names)}")

    # --- Dataset preparation (train/test/validation) ---
    print("[INFO] Preparing processed dataset splits …")
    os.makedirs(args.dataset_out_dir, exist_ok=True)

    # Build consistent class index mapping (class1..classN)
    labels = list(pd.Series(df[args.class_col]).dropna().unique())
    labels_sorted = sorted(labels)
    label_to_index = {lab: i for i, lab in enumerate(labels_sorted)}  # 0..N-1
    class_index_to_name = {i: f"class{i+1}" for i in range(len(labels_sorted))}

    ensure_class_folders(args.dataset_out_dir, class_index_to_name)

    # First split: train vs (test+val)
    train_df, testval_df = train_test_split(
        df, test_size=tv, random_state=args.random_state, stratify=df[args.class_col]
    )
    # Second split: split testval into test and val with correct proportions
    if tv > 0:
        test_fraction_within_tv = args.test_size / tv
        test_df, val_df = train_test_split(
            testval_df, test_size=1 - test_fraction_within_tv,
            random_state=args.random_state, stratify=testval_df[args.class_col]
        )
    else:
        test_df = df.iloc[0:0].copy()
        val_df = df.iloc[0:0].copy()

    # Copy files
    move_matrices(train_df, "train", args.protein_col, args.chain_col, args.class_col,
                  args.sparse_out_dir, args.dataset_out_dir, label_to_index)
    move_matrices(test_df, "test", args.protein_col, args.chain_col, args.class_col,
                  args.sparse_out_dir, args.dataset_out_dir, label_to_index)
    move_matrices(val_df, "validation", args.protein_col, args.chain_col, args.class_col,
                  args.sparse_out_dir, args.dataset_out_dir, label_to_index)

    print("[DONE] Process complete.")


if __name__ == "__main__":
    main()

