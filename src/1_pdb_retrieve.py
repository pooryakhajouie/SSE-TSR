import urllib.request
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import os

# Define input parameters directly in the code
sample_Details = 'sample_details_9k.csv'
out_dir = 'Dataset/'

# Create the output directory
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(sample_Details)
protList = df['protein']
print(len(protList), protList)


def parallelcode(fname):
    print(fname)
    try:
        urllib.request.urlretrieve('http://files.rcsb.org/download/'+fname+'.pdb', out_dir+fname+'.pdb')
    except urllib.error.HTTPError as e:
        if e.code == 404:  # HTTP Error 404: Not Found
            print(f"Skipping {fname} - Protein file not found.")
        else:
            raise  # Re-raise the exception if it's not a 404 error


def getInputs():
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, verbose=50)(delayed(parallelcode)(fname) for fname in protList)


getInputs()

print("code end.")

