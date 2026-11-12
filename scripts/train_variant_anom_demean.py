"""Runs scripts/train_variant.py with demeaned-sea-level anomaly inputs. LAGS (environment variable, default 0) = comma list of lags in days.
Usage: LAGS=0,7,14,30 SEED=1 python scripts/train_variant_anom_demean.py NAME ch1,ch2,...   (channels are sst_l<lag>, sla_l<lag>)"""
import os
import runpy

import oceanembed.dataset as ds
from oceanembed.dataset_demean import AnomDemeanData

LAGS = [int(x) for x in os.environ.get("LAGS", "0").split(",")]
AnomDemeanData.lags = LAGS
ds.Data = AnomDemeanData
ds.CHANNELS = [f"{c}_l{L}" for L in LAGS for c in ("sst", "sla")]
runpy.run_path("scripts/train_variant.py", run_name="__main__")
