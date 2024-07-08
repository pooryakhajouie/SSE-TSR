import os
import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import multiprocessing

with open('sample_details_9k.csv', 'r') as file:
    reader = csv.DictReader(file)
    mapping_df = sorted(reader, key=lambda x: (x['protein'], x['chain']))

# Create dictionaries that map protein names to class labels and chains
chain_dict = {row['protein']: row['chain'] for row in mapping_df}
proteins_name = sorted(list(set(row['protein'] for row in mapping_df)))
# proteins_name = ['1FCY', '1FCZ', '1FD0', '1XAP', '2LBD']
type_columns = ['1_3a1', '2_3a2', '3_3a3', '4_2a11c', '5_2a21c', '6_1a2c ', '7_3b1', '8_3b2', '9_3b3', '10_2b11c', '11_2b21c', '12_1b2c', '13_2a21b', '14_2a11b', '15_2b21a', '16_2b11a', '17_3c', '18_1a1b1c']

# Initialize an empty list to store the loaded keys
unique_keys = []

# Specify the path to the text file containing the unique keys
keys_file_path = 'unique_3D_keys.txt'

# Open the file and read the keys into the list
with open(keys_file_path, 'r') as file:
    for line in file:
        # Remove any leading/trailing whitespace and append the key to the list
        unique_keys.append(int(line.strip()))


def read_3Dkeys_theta30_maxdist35(file_path):
    freq_dict = {}
    with open(file_path, 'r') as file:
        header = next(file).strip().split("\t")  # Read and skip the first line
        type_columns = header[1:]  # Extract the type column headers

        for line in file:
            line = line.strip().split("\t")
            key = int(line[0])

            # Create a sub-dictionary to store frequencies for each type
            type_freqs = {}
            for i, freq in enumerate(line[1:]):
                type_name = type_columns[i]
                type_freqs[type_name] = float(freq)

            freq_dict[key] = type_freqs

    return freq_dict

# Define the function to process a single protein
def process_protein(protein_name, chain_name, file_path, unique_keys):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping protein {protein_name}...")
        return
    freq_dict = read_3Dkeys_theta30_maxdist35(file_path)

    # Create the transformed matrix for the protein
    protein_matrix = []
    for type_column in type_columns:
        row_data = [freq_dict[key][type_column] if key in freq_dict else 0 for key in unique_keys]
        protein_matrix.append(row_data)

    # Convert the matrix to a DataFrame
    df = pd.DataFrame(protein_matrix, columns=unique_keys, index=type_columns)

    # Convert DataFrame to sparse matrix
    sparse_matrix = csr_matrix(df.values)

    # Save the sparse matrix
    output_filename = os.path.join(out_dir, f"{protein_name}_{chain_name}_transformed_sparse_matrix.npz")
    save_npz(output_filename, sparse_matrix)

if __name__ == "__main__":
    out_dir = 'Transformed_sparse_matrices/'
    os.makedirs(out_dir, exist_ok=True)

    # Create a multiprocessing Pool with the number of cores you want to utilize
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # List of arguments for each protein to be processed in parallel
    protein_args = [(protein_name, chain_dict[protein_name], f'Triplet_type/{protein_name}_{chain_dict[protein_name]}.3Dkeys_theta30_maxdist35', unique_keys) for protein_name in proteins_name]

    # Use the multiprocessing Pool to process proteins in parallel
    pool.starmap(process_protein, protein_args)

print("Sparse matrices files created.")
print("number of cores = ", num_cores)
