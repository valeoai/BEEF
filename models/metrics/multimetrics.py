import torch
import torch.nn as nn

from .map import mAP
from .futuretraj import FutureTraj

METRICS_TO_CLS = {'map': mAP, 'future_traj':FutureTraj}

class MultiMetrics(nn.Module):
    def __init__(self, engine=None, mode=None, metrics=[]):
        super(MultiMetrics, self).__init__()
        self.engine = engine
        self.mode = mode
        self.metrics = []
        for metric in metrics:
            self.metrics.append(METRICS_TO_CLS[metric](self.engine, mode=self.mode))

    def forward(self, cri_out, net_out, batch):
        out_metrics = {}
        for metric in self.metrics:
            out_metrics.update(metric(cri_out, net_out, batch))
        return out_metrics
