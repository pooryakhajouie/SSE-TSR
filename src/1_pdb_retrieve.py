import os
import argparse
import urllib.request
import urllib.error
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path
import time

# Fixed defaults (no CLI flags for these to keep it simple)
RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{code}.pdb"
RETRIES = 2
RETRY_WAIT = 1.0  # seconds

def parse_args():
    p = argparse.ArgumentParser(
        description="Download PDB files from RCSB for protein codes listed in a CSV (expects a 'protein' column)."
    )
    p.add_argument("sample_details", help="Input CSV with a 'protein' column.")
    p.add_argument("out_dir", help="Directory to save downloaded PDB files.")
    p.add_argument("-o", "--cleaned-csv", default=None,
                   help="Path for cleaned CSV (rows with missing PDBs removed). "
                        "Default: <input_stem>_cleaned.csv")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                   help="Number of parallel downloads (default: CPU count).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download and overwrite existing files.")
    return p.parse_args()

def try_download(code, out_dir, overwrite):
    url = RCSB_URL_TEMPLATE.format(code=code)
    dest = os.path.join(out_dir, f"{code}.pdb")

    if not overwrite and os.path.exists(dest):
        return code, True

    os.makedirs(out_dir, exist_ok=True)

    attempt = 0
    while True:
        try:
            urllib.request.urlretrieve(url, dest)
            return code, True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  404 Not Found: {code}")
                return code, False
            attempt += 1
            if attempt > RETRIES:
                print(f"  HTTP {e.code} for {code} after {RETRIES} retries; giving up.")
                return code, False
            time.sleep(RETRY_WAIT)
        except Exception as e:
            attempt += 1
            if attempt > RETRIES:
                print(f"  Error downloading {code} after {RETRIES} retries: {e}")
                return code, False
            time.sleep(RETRY_WAIT)

def main():
    args = parse_args()

    # Decide cleaned CSV path
    if args.cleaned_csv is None:
        inp = Path(args.sample_details)
        args.cleaned_csv = str(inp.with_name(f"{inp.stem}_cleaned.csv"))

    # Load CSV and get protein codes
    df = pd.read_csv(args.sample_details)
    if "protein" not in df.columns:
        raise ValueError(f"Column 'protein' not found in {args.sample_csv}. Available: {list(df.columns)}")

    prot_list = df["protein"].astype(str).str.upper().tolist()
    print(f"Attempting to download {len(prot_list)} PDBs to '{args.out_dir}'")

    # Parallel download
    jobs = max(1, int(args.jobs or 1))
    results = Parallel(n_jobs=jobs, verbose=10)(
        delayed(try_download)(code, args.out_dir, args.overwrite) for code in prot_list
    )

    # Collect failures and write cleaned CSV
    failed = [code for code, ok in results if not ok]
    print(f"\nTotal failures (404 or error): {len(failed)}")

    if failed:
        before = len(df)
        df_clean = df[~df["protein"].astype(str).str.upper().isin(set(failed))].copy()
        after = len(df_clean)
        df_clean.to_csv(args.cleaned_csv, index=False)
        print(f"Dropped {before - after} entries; cleaned CSV saved as '{args.cleaned_csv}'")
    else:
        print("No failures; original CSV is unchanged.")

    print("Download step complete.")

if __name__ == "__main__":
    main()

