"""Prediction with a frozen model set (configs/final_model.yaml: pipeline anom; configs/final_model_demean.yaml: pipeline demean) for any list of days.
Pure inference: no targets are read here."""
import numpy as np
import torch
import yaml

from .dataset import TARGET_SCALE
from .dataset_anom import AnomData
from .dataset_demean import AnomDemeanData
from .final_model import UNet, device, lag_inputs

__all__ = ["ModelSet", "device"]


class ModelSet:
    def __init__(self, cfg_path, dev):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.dev = dev
        self.pipeline = self.cfg.get("pipeline", "anom")
        if self.pipeline == "anom":
            self.D = AnomData()
            self.D.fit_input_clim()
            self.base = self.D
        else:
            self.Du, self.Dl = AnomDemeanData(), AnomDemeanData()
            self.Du.lags, self.Dl.lags = [0], [0, 7, 14, 30]
            for d in (self.Du, self.Dl):
                d.fit_input_clim()
                d.fit_demean_std()
            self.base = self.Du
        self.members = []
        for m in self.cfg["members"]:
            if self.pipeline == "anom":
                lagged = "lags" in m["file"]
                cin = 9 if lagged else 3
            else:
                lagged = len(m["lags"]) > 1
                cin = 2 * len(m["lags"]) + 1
            net = UNet(cin).to(dev)
            net.load_state_dict(torch.load(m["file"], map_location=dev))
            self.members.append((lagged, net.eval()))

    def _inputs(self, lagged, day):
        if self.pipeline == "anom":
            return lag_inputs(self.D, day) if lagged else self.D.inputs(day)[[0, 1, 4]]
        return (self.Dl if lagged else self.Du).inputs(day)

    @torch.no_grad()
    def predict_anomaly(self, days):
        """Ensemble-mean GLORYS-anomaly prediction in C, shape (len(days), 15, 101, 241)."""
        Xu = torch.from_numpy(np.stack([self._inputs(False, d) for d in days])).to(self.dev)
        Xl = torch.from_numpy(np.stack([self._inputs(True, d) for d in days])).to(self.dev)
        out = sum(net(Xl if lagged else Xu) for lagged, net in self.members) / len(self.members)
        return (TARGET_SCALE * out).cpu().numpy()
