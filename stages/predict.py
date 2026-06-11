#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: Prediction / Inference â€” score compounds with a trained model.

Usage:
    python -m stages.predict --model models/model.pth --input data/pool.csv --output predictions.csv
    python -m stages.predict --model models/model.pth --input data/pool.csv --mc-samples 20
    python -m stages.predict --model models/model.pth --input data/pool.csv --gpu-ids 0,1,2,3

Parameters:
    --model         Path to trained model .pth file (required)
    --input         CSV with SMILES to score (required)
    --output        Output CSV with predictions (default: output/predictions.csv)
    --smiles-col    SMILES column name (default: "SMILES")
    --batch-size    Inference batch size (default: 256)
    --mc-samples    MC Dropout forward passes for uncertainty (default: 10)
    --device        Device: cpu, cuda, cuda:0 (default: auto)
    --gpu-ids       Comma-separated GPU IDs for multi-GPU (e.g. "0,1,2,3")
    --streaming     Enable streaming mode for very large datasets (flag)
    --chunk-size    Streaming chunk size (default: 100000)
    --top-k         Only output top K predictions (default: all)
"""

import argparse
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


def run_predict(
    model_path: str,
    input_file: str,
    output_file: str = "output/predictions.csv",
    smiles_col: str = "SMILES",
    batch_size: int = 256,
    mc_samples: int = 10,
    device: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    streaming: bool = False,
    chunk_size: int = 100000,
    top_k: Optional[int] = None,
) -> str:
    """Run inference on a pool of compounds.

    Args:
        model_path: Path to trained model checkpoint.
        input_file: CSV with SMILES.
        output_file: Output CSV path.
        smiles_col: Column with SMILES strings.
        batch_size: Inference batch size.
        mc_samples: Number of MC Dropout forward passes.
        device: Compute device.
        gpu_ids: List of GPU IDs for multi-GPU inference.
        streaming: Use streaming mode for large datasets.
        chunk_size: Chunk size for streaming mode.
        top_k: Only keep top K predictions.

    Returns:
        Path to output CSV.
    """
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")

    from siamese_GNN.featurization import feature_dims
    from siamese_GNN.model import SiameseRankNet

    # Model checkpoint is a raw state_dict (saved by train_ranknet)
    # Infer dimensions from feature_dims helper
    node_dim, edge_dim = feature_dims()

    # Determine hidden_dim from state dict
    state_dict = checkpoint if isinstance(checkpoint, dict) and "encoder.node_lin.weight" in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    if "encoder.node_lin.weight" in state_dict:
        inferred_hidden = state_dict["encoder.node_lin.weight"].shape[0]
    else:
        inferred_hidden = 64

    # Determine dropout from saved config if available
    inferred_dropout = 0.3

    model = SiameseRankNet(
        node_in=node_dim,
        edge_in=edge_dim,
        hidden_dim=inferred_hidden,
        dropout=inferred_dropout,
    )
    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_path} (hidden_dim={inferred_hidden})")

    # Resolve device
    if gpu_ids and len(gpu_ids) > 1:
        # Multi-GPU inference
        from inference.multigpu import MultiGPUPredictor
        print(f"Multi-GPU inference on GPUs: {gpu_ids}")
        inferrer = MultiGPUPredictor(model=model, gpu_ids=gpu_ids)
        device = None
    else:
        if device is None:
            if gpu_ids:
                device = f"cuda:{gpu_ids[0]}"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        inferrer = None
        print(f"Device: {device}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    if streaming and inferrer:
        # Multi-GPU streaming via predict_file
        total = inferrer.predict_file(
            input_file, output_file, batch_size=batch_size,
            chunk_size=chunk_size, smiles_col=0,
        )
        print(f"Multi-GPU streaming: {total} compounds â†’ {output_file}")
    elif streaming:
        # Single-device streaming
        print(f"Streaming mode (chunk_size={chunk_size})...")
        _predict_streaming(
            model, device, input_file, output_file,
            smiles_col, batch_size, mc_samples, chunk_size, top_k,
        )
    elif inferrer:
        # Multi-GPU standard (no MC Dropout â€” single forward pass)
        df = pd.read_csv(input_file)
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not in {list(df.columns)}")
        smiles_list = df[smiles_col].tolist()
        print(f"Scoring {len(smiles_list)} compounds (multi-GPU)...")
        start = time.time()
        scores, valid_mask = inferrer.predict(smiles_list, batch_size=batch_size)
        elapsed = time.time() - start
        print(f"Inference done in {elapsed:.1f}s")
        results = df.copy()
        results["predicted_score"] = scores
        results["uncertainty"] = 0.0
        results = results.sort_values("predicted_score", ascending=True)
        if top_k:
            results = results.head(top_k)
        results.to_csv(output_file, index=False)
        print(f"Saved {len(results)} predictions to {output_file}")
    else:
        # Standard single-device mode with MC Dropout
        df = pd.read_csv(input_file)
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not in {list(df.columns)}")

        smiles_list = df[smiles_col].tolist()
        print(f"Scoring {len(smiles_list)} compounds (MC samples={mc_samples})...")
        start = time.time()

        predictions, uncertainties, valid_mask = _predict_batch(
            model, inferrer, device, smiles_list, batch_size, mc_samples
        )

        elapsed = time.time() - start
        print(f"Inference done in {elapsed:.1f}s "
              f"({len(smiles_list)/elapsed:.0f} compounds/s)")

        # Build output
        results = df.copy()
        results["predicted_score"] = np.nan
        results["uncertainty"] = np.nan
        results.loc[valid_mask, "predicted_score"] = predictions
        results.loc[valid_mask, "uncertainty"] = uncertainties

        # Sort by predicted score (lower = better for docking)
        results = results.sort_values("predicted_score", ascending=True)

        if top_k:
            results = results.head(top_k)
            print(f"Keeping top {top_k} predictions")

        results.to_csv(output_file, index=False)
        print(f"Saved {len(results)} predictions to {output_file}")

    return output_file


def _predict_batch(model, inferrer, device, smiles_list, batch_size, mc_samples):
    """Run MC Dropout prediction on a list of SMILES."""
    from siamese_GNN.featurization import smiles_to_graph_gine
    from siamese_GNN.model import enable_mc_dropout
    from torch_geometric.data import Batch

    # Featurize
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph_gine(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)

    valid_mask = [i in set(valid_indices) for i in range(len(smiles_list))]

    if not graphs:
        return [], [], valid_mask

    # MC Dropout inference: eval mode + dropout enabled
    enable_mc_dropout(model)

    all_scores = []
    with torch.no_grad():
        for _ in range(mc_samples):
            batch_scores = []
            for i in range(0, len(graphs), batch_size):
                batch = Batch.from_data_list(graphs[i:i + batch_size])
                batch = batch.to(device)
                scores = model.forward_one(batch).cpu().numpy()
                batch_scores.extend(scores.flatten())
            all_scores.append(batch_scores)

    all_scores = np.array(all_scores)  # (mc_samples, n_compounds)
    predictions = all_scores.mean(axis=0)
    uncertainties = all_scores.std(axis=0)

    return predictions, uncertainties, valid_mask


def _predict_streaming(model, device, input_file, output_file,
                       smiles_col, batch_size, mc_samples, chunk_size, top_k):
    """Stream prediction for very large datasets."""

    first_chunk = True
    total_processed = 0

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        if smiles_col not in chunk.columns:
            raise ValueError(f"Column '{smiles_col}' not in {list(chunk.columns)}")

        smiles_list = chunk[smiles_col].tolist()
        predictions, uncertainties, valid_mask = _predict_batch(
            model, None, device, smiles_list, batch_size, mc_samples
        )

        chunk["predicted_score"] = np.nan
        chunk["uncertainty"] = np.nan
        chunk.loc[valid_mask, "predicted_score"] = predictions
        chunk.loc[valid_mask, "uncertainty"] = uncertainties

        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        first_chunk = False
        total_processed += len(chunk)
        print(f"  Processed {total_processed} compounds...")

    # If top_k, re-read and filter
    if top_k:
        df = pd.read_csv(output_file)
        df = df.sort_values("predicted_score", ascending=True).head(top_k)
        df.to_csv(output_file, index=False)
        print(f"Filtered to top {top_k}")

    print(f"Streaming complete: {total_processed} compounds â†’ {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 4: Model Inference / Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.predict --model models/model.pth --input data/pool.csv
  python -m stages.predict --model models/model.pth --input data/pool.csv --mc-samples 20
  python -m stages.predict --model models/model.pth --input data/big_lib.csv --streaming
  python -m stages.predict --model models/model.pth --input pool.csv --gpu-ids 0,1,2,3
  python -m stages.predict --model models/model.pth --input pool.csv --top-k 1000
        """,
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Trained model .pth file")
    parser.add_argument("--input", "-i", required=True,
                        help="CSV with SMILES to score")
    parser.add_argument("--output", "-o", default="output/predictions.csv",
                        help="Output predictions CSV")
    parser.add_argument("--smiles-col", default="SMILES",
                        help="SMILES column name (default: SMILES)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size (default: 256)")
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="MC Dropout forward passes (default: 10)")
    parser.add_argument("--device", help="Device: cpu, cuda, cuda:0 (default: auto)")
    parser.add_argument("--gpu-ids", help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--streaming", action="store_true",
                        help="Streaming mode for large datasets")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Streaming chunk size (default: 100000)")
    parser.add_argument("--top-k", type=int,
                        help="Only output top K predictions")
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    run_predict(
        model_path=args.model,
        input_file=args.input,
        output_file=args.output,
        smiles_col=args.smiles_col,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        device=args.device,
        gpu_ids=gpu_ids,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
