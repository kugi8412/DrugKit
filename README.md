# DrugKit
A Python library for drug screening using the Deep Docking protocol.

## DrugKit: GNN-Powered Deep Docking Protocol
DrugKit is an open-source Python framework designed to accelerate large-scale virtual screening for structure-based drug discovery. Taking inspiration from the Deep Docking methodology, DrugKit replaces traditional QSAR molecular fingerprints with cutting-edge Graph Neural Networks (GNNs). This allows it to learn complex topological representations of molecules and approximate rigid docking scores across billions of compounds in a fraction of the time. The solution is based on the [publication](https://pubs.acs.org/doi/10.1021/acscentsci.0c00229), along with the code available in the [repository](https://github.com/jamesgleave/DD_protocol).

## Overview
Traditional structure-based virtual screening  of ultra-large libraries (like the 1B+ compound ZINC database) requires immense computational resources. Deep Docking solves this by:

1. Docking a small, representative subset of the library with possibility to use prior knowledge of knowing ligands.
2. Training a deep learning model to predict docking scores based on molecular structure.
3. Rapidly inferring the scores for the rest of the billion-scale library.

DrugKit modernizes this approach by treating molecules as graphs (atoms as nodes, bonds as edges). This solution allows to capture richer spatial and chemical features, leading to higher hit-enrichment rates than standard Morgan fingerprint-based Multi-Layer Perceptrons (MLPs).

## Task assigment

Phase 1: Data Processing

    [OK] Integration with RDKit to parse SMILES strings improve some functions.

    [OK] Conversion of SMILES into PyTorch Geometric.

    [OK] Node featurization (atom type, hybridization, formal charge, chirality).

    [OK] Edge featurization (bond type, stereochemistry, ring status).

    [OK] API for different methods to pocekt detection.

Phase 2: The Deep Docking Pipeline

    [OK] Sampler Module: Based on Tanimoto Clustering.

    [OK] Basic wrapper for Autodock Vina.

    [OK] Docking Wrapper: Seamless API integration with popular docking engines (Glide, Smina) to generate ground-truth training scores.

    [OK] Find Testing data to compare with original Deep Docking Protocol.

    [OK] GNN Model Architectures and make ablations for example data.

    [OK] Iterative Active Learning: An iterative loop that trains the GNN, predicts scores, and requests exact docking for high-uncertainty (Monte Carlo Dropout).

Phase 3: Billion-Scale Inference

    [OK] Batch processing pipeline for massive SMILES datasets.

    [OK] Multi-GPU inference support.

## Installation

### From TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ drugkit
```

### Development (editable)

```bash
conda env create -f config/drugkit.yaml -n drugkit
conda activate drugkit
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```
