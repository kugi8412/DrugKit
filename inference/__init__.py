#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Billion-scale inference: batch processing and multi-GPU support.
"""


from inference.batch_inference import batch_predict, batch_predict_from_file
from inference.multigpu import MultiGPUPredictor

__all__ = [
    "batch_predict",
    "batch_predict_from_file",
    "MultiGPUPredictor",
]
