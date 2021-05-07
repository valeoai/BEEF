import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from bootstrap.lib.logger import Logger


class FutureTraj(nn.Module):
    def __init__(self, engine=None, mode=None):
        super(FutureTraj, self).__init__()
        self.engine = engine
        self.mode = mode
        self.fps = self.engine.dataset[self.mode].fps

    def forward(self, cri_out, net_out, batch):
        next_xy = batch['r_next_xy'] # b x 6 x 2
        gt_xy = next_xy
        bsize = next_xy.size(0)
        n_future = next_xy.size(1)

        if 'pred_xy' in net_out:
            pred_xy = net_out['pred_xy'][:,-n_future:,:]
            assert pred_xy.size(1) == n_future
        else:
            raise KeyError

        diff = pred_xy - gt_xy

        # loss_values = self.loss(pred_next_xy, next_xy)
        loss_values = torch.sqrt(torch.sum((diff)**2, -1))

        loss_value = loss_values.mean()

        out = {'future_mse': loss_value}

        # n_frames x n_classes
        return out
