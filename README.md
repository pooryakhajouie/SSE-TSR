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
