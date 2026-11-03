"""Runs scripts/train_variant.py with anomaly inputs (oceanembed.dataset_anom.AnomData) instead of raw inputs.
Usage: python scripts/train_variant_anom.py NAME ch1,ch2,..."""
import runpy

import oceanembed.dataset as ds
from oceanembed.dataset_anom import AnomData

ds.Data = AnomData
runpy.run_path("scripts/train_variant.py", run_name="__main__")
