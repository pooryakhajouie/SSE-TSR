import os
import urllib.request
import urllib.error
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd

# ——— CONFIG ———
sample_details_csv = 'sample_details_functional.csv'
out_dir            = 'Dataset/'
cleaned_csv        = 'sample_details_functional_cleaned.csv'

os.makedirs(out_dir, exist_ok=True)

# ——— LOAD CSV ———
df = pd.read_csv(sample_details_csv)
prot_list = df['protein'].astype(str).str.upper().tolist()
print(f"Attempting to download {len(prot_list)} PDBs")

# ——— DOWNLOAD FUNCTION ———
def try_download(code):
    """
    Attempts to download {code}.pdb from RCSB. 
    Returns (code, True) on success, (code, False) if 404, re-raises otherwise.
    """
    url = f'https://files.rcsb.org/download/{code}.pdb'
    dest= os.path.join(out_dir, f'{code}.pdb')
    if os.path.exists(dest):
        return code, True
    try:
        urllib.request.urlretrieve(url, dest)
        return code, True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # File genuinely not present on the RCSB server
            print(f"  404 Not Found: {code}")
            return code, False
        else:
            # Some other HTTP error (500, timeout, etc.)
            raise
    except Exception as e:
        # Any other issue (network, disk, etc.)
        print(f"  Error downloading {code}: {e}")
        return code, False

# ——— PARALLEL DOWNLOAD ———
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, verbose=10)(
    delayed(try_download)(code) for code in prot_list
)

# ——— COLLECT FAILURES ———
failed = [code for code, ok in results if not ok]
print(f"\nTotal failures (404 or error): {len(failed)}")

# ——— FILTER OUT FAILED FROM DATAFRAME ———
if failed:
    before = len(df)
    df_clean = df[~df['protein'].str.upper().isin(failed)].copy()
    after  = len(df_clean)
    df_clean.to_csv(cleaned_csv, index=False)
    print(f"Dropped {before-after} entries; cleaned CSV saved as '{cleaned_csv}'")
else:
    print("No failures; original CSV is unchanged.")

print("Download step complete.")

