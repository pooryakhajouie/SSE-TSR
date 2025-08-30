import os
import csv
import argparse
import multiprocessing

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect unique 3D key IDs across proteins from per-chain .3Dkeys files."
    )
    p.add_argument("sample_details",
                   help="CSV with at least columns: 'protein' and 'chain' (default names can be changed).")
    p.add_argument("data_dir",
                   help="Directory containing the per-chain files (e.g., Dataset/lexicographic).")
    p.add_argument("-o", "--output", default="unique_3D_keys.txt",
                   help="Path to write the unique keys list (default: unique_3D_keys.txt).")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                   help="Number of worker processes (default: CPU count).")
    p.add_argument("--filename-template",
                   default="{protein}_{chain}.3Dkeys_theta30_maxdist35",
                   help="Relative filename template under data_dir. "
                        "Use placeholders {protein} and {chain}. "
                        "Default: {protein}_{chain}.3Dkeys_theta30_maxdist35")
    p.add_argument("--delimiter", default=",",
                   help="CSV delimiter (default: ',').")
    p.add_argument("--protein-col", default="protein",
                   help="Column name for protein in mapping CSV (default: 'protein').")
    p.add_argument("--chain-col", default="chain",
                   help="Column name for chain in mapping CSV (default: 'chain').")
    p.add_argument("--all-chains", action="store_true",
                   help="Process every (protein, chain) pair in the CSV (instead of one chain per protein).")
    return p.parse_args()

def read_3Dkeys_file(file_path):
    """Reads a .3Dkeys_* file and returns a list of integer keys (first column)."""
    keys = []
    with open(file_path, "r") as f:
        header = next(f)  # skip header line
        for line in f:
            if not line.strip():
                continue
            # Only need the first column (key)
            first_field = line.split("\t", 1)[0]
            try:
                keys.append(int(first_field))
            except ValueError:
                # Skip malformed rows
                continue
    return keys

def process_protein(protein_name, chain_name, data_dir, template):
    file_rel = template.format(protein=protein_name, chain=chain_name)
    file_path = os.path.join(data_dir, file_rel)
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}. Skipping...")
        return []
    return read_3Dkeys_file(file_path)

def main():
    args = parse_args()

    # Read mapping CSV
    with open(args.sample_details, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=args.delimiter)
        fieldnames = set(reader.fieldnames or [])
        required = {args.protein_col, args.chain_col}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        rows = list(reader)

    if args.all_chains:
        # Use every unique (protein, chain) pair
        protein_chain_pairs = sorted({(r[args.protein_col], r[args.chain_col]) for r in rows})
    else:
        # Map each protein to a single chain (last occurrence wins, same as your original logic)
        chain_dict = {}
        for r in rows:
            chain_dict[r[args.protein_col]] = r[args.chain_col]
        protein_chain_pairs = sorted(chain_dict.items())  # [(protein, chain), ...]

    if not protein_chain_pairs:
        print("[INFO] No protein/chain pairs found in the mapping CSV. Nothing to do.")
        return

    # Parallel processing
    jobs = max(1, int(args.jobs or 1))
    with multiprocessing.Pool(processes=jobs) as pool:
        work = [(p, c, args.data_dir, args.filename_template) for (p, c) in protein_chain_pairs]
        result_lists = pool.starmap(process_protein, work)

    # Combine and uniquify
    all_keys = []
    for keys in result_lists:
        all_keys.extend(keys)
    unique_keys = sorted(set(all_keys))

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for k in unique_keys:
            f.write(f"{k}\n")

    print(f"[DONE] Wrote {len(unique_keys)} unique keys to: {args.output}")

if __name__ == "__main__":
    main()

