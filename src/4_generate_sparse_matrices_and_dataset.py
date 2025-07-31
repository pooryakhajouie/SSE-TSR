import os
import csv
import shutil
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split
import multiprocessing

TYPE_COLUMNS = ['1_3a1', '2_3a2', '3_3a3', '4_2a11c', '5_2a21c', '6_1a2c',
                '7_3b1', '8_3b2', '9_3b3', '10_2b11c', '11_2b21c', '12_1b2c',
                '13_2a21b', '14_2a11b', '15_2b21a', '16_2b11a', '17_3c',
                '18_1a1b1c']


def read_keys_file(file_path):
    freq_dict = {}
    with open(file_path, 'r') as handle:
        header = next(handle).strip().split("\t")
        types = header[1:]
        for line in handle:
            parts = line.strip().split("\t")
            key = int(parts[0])
            freq_dict[key] = {types[i]: float(freq) for i, freq in enumerate(parts[1:])}
    return freq_dict


def process_protein(protein_name, chain_name, file_path, unique_keys, out_dir):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping {protein_name}...")
        return
    freq_dict = read_keys_file(file_path)
    matrix = []
    for col in TYPE_COLUMNS:
        row = [freq_dict[key][col] if key in freq_dict else 0 for key in unique_keys]
        matrix.append(row)
    df = pd.DataFrame(matrix, columns=unique_keys, index=TYPE_COLUMNS)
    sparse = csr_matrix(df.values)
    save_npz(os.path.join(out_dir, f"{protein_name}_{chain_name}_transformed_sparse_matrix.npz"), sparse)


def generate_sparse_matrices(mapping, unique_keys, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    args = [(row['protein'], row['chain'],
             f"Dataset/lexicographic/{row['protein']}_{row['chain']}.3Dkeys_theta30_maxdist35",
             unique_keys, out_dir) for row in mapping]
    pool.starmap(process_protein, args)
    print("Sparse matrix files created.")
    print("number of cores =", num_cores)


def prepare_dataset(mapping_df, dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for folder in ["train", "test", "validation"]:
        fp = os.path.join(output_path, folder)
        os.makedirs(fp, exist_ok=True)
        for cls in ["class1", "class2", "class3"]:
            os.makedirs(os.path.join(fp, cls), exist_ok=True)

    train_df, test_val_df = train_test_split(mapping_df, test_size=0.2, random_state=42, stratify=mapping_df['group1'])
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df['group1'])

    def move_split(df_split, folder):
        for _, row in df_split.iterrows():
            protein_name = row['protein']
            protein_chain = row['chain']
            protein_class = row['group1']
            matrix_name = f"{protein_name}_{protein_chain}_transformed_sparse_matrix.npz"
            src = os.path.join(dataset_path, matrix_name)
            dst = os.path.join(output_path, folder, f"class{protein_class + 1}", matrix_name)
            if not os.path.exists(src):
                print(f"File not found: {src}. Skipping {protein_name}...")
                continue
            shutil.copy(src, dst)

    move_split(train_df, "train")
    move_split(test_df, "test")
    move_split(val_df, "validation")


def main():
    with open('sample_details_9k.csv', 'r') as f:
        mapping = list(csv.DictReader(f))
    unique_keys = [int(line.strip()) for line in open('unique_3D_keys.txt')]

    generate_sparse_matrices(mapping, unique_keys, 'Transformed_sparse_matrices/')
    mapping_df = pd.DataFrame(mapping)
    prepare_dataset(mapping_df, 'Transformed_sparse_matrices/', 'processed_dataset/')


if __name__ == '__main__':
    main()
