import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = "Transformed_sparse_matrices/"
output_path = "processed_dataset/"

os.makedirs(output_path, exist_ok=True)

# Load dataset details from CSV
df = pd.read_csv("sample_details_9k.csv")

# Create train, test, and validation folders
for folder in ["train", "test", "validation"]:
    folder_path = os.path.join(output_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create class folders inside each train, test, and validation folder
    for protein_class in ["class1", "class2", "class3"]:
        class_folder_path = os.path.join(folder_path, protein_class)
        os.makedirs(class_folder_path, exist_ok=True)

# Split dataset while preserving class distribution
train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['group1'])
test_df, validation_df = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df['group1'])

# Move images to the corresponding folders
def move_images(df, folder):
    for index, row in df.iterrows():
        protein_name = row['protein']
        protein_chain = row['chain']
        protein_class = row['group1']
        matrix_name = f"{protein_name}_{protein_chain}_transformed_sparse_matrix.npz"
        
        source_path = os.path.join(dataset_path, matrix_name)
        destination_path = os.path.join(output_path, folder, f"class{protein_class + 1}", matrix_name)
        if not os.path.exists(source_path):
            print(f"File not found: {source_path}. Skipping protein {protein_name}...")
            continue
        shutil.copy(source_path, destination_path)

# Move images to train, test, and validation folders
move_images(train_df, "train")
move_images(test_df, "test")
move_images(validation_df, "validation")
