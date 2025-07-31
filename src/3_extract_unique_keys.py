import os
import csv
import numpy as np
import multiprocessing

with open('sample_details_9k.csv', 'r') as file:
    reader = csv.DictReader(file)
    mapping_df = sorted(reader, key=lambda x: (x['protein'], x['chain']))

# Create dictionaries that map protein names to class labels and chains
chain_dict = {row['protein']: row['chain'] for row in mapping_df}
proteins_name = sorted(list(set(row['protein'] for row in mapping_df)))

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

# Define a function to process a single protein
def process_protein(protein_name, chain_name):
    file_path = f'Dataset/lexicographic/{protein_name}_{chain_name}.3Dkeys_theta30_maxdist35'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping...")
        return []
    freq_dict = read_3Dkeys_theta30_maxdist35(file_path)
    keys = list(freq_dict.keys())
    return keys

if __name__ == "__main__":
    # Create a multiprocessing Pool with the number of cores you want to utilize
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # List of arguments for each protein to be processed in parallel
    protein_args = [(protein_name, chain_dict[protein_name]) for protein_name in proteins_name]

    # Use the multiprocessing Pool to process proteins in parallel
    result_lists = pool.starmap(process_protein, protein_args)

    # Combine the results from all processes into a single list
    all_keys = []
    for keys in result_lists:
        all_keys.extend(keys)

    # Get the unique keys
    unique_keys = sorted(set(all_keys))

    # Save unique_keys to a text file as shown in the previous code snippet
    with open('unique_3D_keys.txt', 'w') as file:
        for key in unique_keys:
            file.write(str(key) + '\n')
    print(f"Length of unique_keys: {len(unique_keys)}")
