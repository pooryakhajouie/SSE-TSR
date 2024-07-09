# SSE-TSR

This project is designed for secondary structure classification of proteins. The workflow consists of several Python scripts that need to be executed in a specific order, each performing a distinct step in the process. Below is a detailed description of each step and how to use the scripts.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Retrieve PDB Files](#step-1-retrieve-pdb-files)
  - [Step 2: Key Transformation](#step-2-key-transformation)
  - [Step 3: Copy 3D Keys](#step-3-copy-3d-keys)
  - [Step 4: Extract Unique Keys](#step-4-extract-unique-keys)
  - [Step 5: Generate Sparse Matrices](#step-5-generate-sparse-matrices)
  - [Step 6: Dataset Preparation](#step-6-dataset-preparation)
  - [Step 7: Secondary Structure Classification](#step-7-secondary-structure-classification)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project involves the following steps:

1. Retrieving PDB files from the PDB bank.
2. Transforming keys from 1D and 3D structures.
3. Copying 3D keys with specific parameters.
4. Extracting unique keys from the dataset.
5. Generating sparse matrices for each protein.
6. Preparing the dataset into training, testing, and validation sets.
7. Training and evaluating a model for secondary structure classification.

## Prerequisites

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `biopython`
- Internet connection to retrieve PDB files

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PooryaKhajoui/SSE-TSR.git
   cd SSE-TSR
2. Install the required packages:
   ```python
   pip install -r requirements.txt
3. Place the following files in the root directory:
    `sample_details.csv`
    `aminoAcidCode_lexicographic_new.txt`
    `amino_codes.txt`

## Usage

1. Step 1: Retrieve PDB Files
   Run the script to retrieve PDB files:
   ```python
   python src/1_pdb_retrieve.py
   ```
   This script requires sample_details.csv, which contains the protein name, chain, and corresponding label.

2. Step 2: Key Generation
   Generate keys from the Ca atoms of each protein file (pdb file). It also categorizes each key into one of the eighteen secondary structure types.
   ```python
   python src/python src/2_keyTransformation1Dand3andHelixSheetNone.py
   ```
   This script uses `sample_details.csv`, `aminoAcidCode_lexicographic_new.txt`, and `amino_codes.txt` files.

3. Step 3: Pick out 3D key files
   Copy 3D keys with specified parameters:
    ```python
   python src/3_copy_3Dkeys_theta30_maxdist35_files.py
   ```
    
4. Step 4: Extract Unique Keys
   Extract unique keys from the dataset:
   ```python
   python src/4_extract_unique_keys.py
   ```
   
5. Step 5: Generate Sparse Matrices
   Generate sparse 2D matrices for each protein:
   ```python
   python src/5_generate_sparse_matrices.py
   ```

6. Step 6: Dataset Preparation
   Prepare the dataset into training, testing, and validation sets:
   ```python
   python src/6_dataset_preparation.py
   ```

7. Step 7: Protein Classification based on their Secondary Structure information
   Load the data with data generators, train and evaluate the model:
   ```python
   python src/7_secondary_structure_classification.py
   ```
This script contains data generators to load the data and train and evaluate the model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

