# SSE-TSR

This project is designed for the classification of proteins using the TSR-based method and the secondary structure information of proteins. The workflow consists of several Python scripts that need to be executed in a specific order, each performing a distinct step in the process. Below is a detailed description of each step and how to use the scripts.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Retrieve PDB Files](#step-1-retrieve-pdb-files)
  - [Step 2: Key Generation](#step-2-key-transformation)
  - [Step 3: Extract Unique Keys](#step-3-extract-unique-keys)
  - [Step 4: Generate Sparse Matrices and Prepare Dataset](#step-4-generate-sparse-matrices-and-prepare-dataset)
  - [Step 5: Protein Classification](#step-5-secondary-structure-classification)

## Overview

This project involves the following steps:

1. Retrieving PDB files from the PDB bank.
2. Generating 3D keys and categorizing them based on their secondary structure information.
3. Extracting unique keys from the dataset.
4. Generating sparse matrices and preparing the dataset.
5. Loading, training, and evaluating the model for protein classification using SSE-TSR keys.

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
   git clone https://github.com/pooryakhajouie/SSE-TSR.git
   cd SSE-TSR
2. Install the required packages:
   ```python
   pip install -r requirements.txt
3. Place the following files in the root directory:
    `sample_details.csv`
    `aminoAcidCode_lexicographic_new.txt`
    `amino_codes.txt`

## Usage

### 1. Step 1: Retrieve PDB Files
   Run the script to retrieve PDB files:
   ```python
   python src/1_pdb_retrieve.py
   ```
   This script requires sample_details.csv, which contains the protein name, chain, and corresponding label.

### 2. Step 2: Key Generation
   Generate keys from the Ca atoms of each protein file (pdb file). It also categorizes each key into one of the eighteen secondary structure types.
   ```python
   python src/python src/2_keyTransformation1D_3D_HelixSheetNone.py
   ```
   This script uses `sample_details.csv`, `aminoAcidCode_lexicographic_new.txt`, and `amino_codes.txt` files.
    
### 3. Step 3: Extract Unique Keys
   Extract unique keys from the dataset:
   ```python
   python src/3_extract_unique_keys.py
   ```
   
### 4. Step 4: Generate Sparse Matrices and Prepare Dataset
   Generate sparse matrices for each protein and split the dataset:
   ```python
   python src/4_generate_sparse_matrices_and_dataset.py
   ```

### 5. Step 5: Protein Classification based on their Secondary Structure information
   Load the data with data generators, train and evaluate the model:
   ```python
   python src/5_secondary_structure_classification.py
   ```
This script contains data generators to load the data and train and evaluate the model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

