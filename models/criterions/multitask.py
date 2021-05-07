import torch
import torch.nn as nn
from .frames_classif import FramesClassifLoss
from .l2_points import L2Points


class MultiTaskHDD(nn.Module):
    def __init__(self,
                 future_only=True,
                 class_freq=None,
                 alpha_dict={}):
        super(MultiTaskHDD, self).__init__()
        self.traj_loss = L2Points()
        self.frames_classif = FramesClassifLoss(class_freq)
        self.alpha_dict = alpha_dict

    def forward(self, net_out, batch):
        if batch['r_prev_xy'].dim() == 4:
            bsize = batch['r_prev_xy'].size(0)
            win = batch['r_prev_xy'].size(1)
            batch['r_prev_xy'] = batch['r_prev_xy'].view(bsize*win, -1, 2)
            batch['r_next_xy'] = batch['r_next_xy'].view(bsize*win, -1, 2)
        out_traj = self.traj_loss(net_out, batch)
        out_classif = self.frames_classif(net_out, batch)
        out = {
        "loss_driving":out_traj['loss'],
        "loss_classif":out_classif['loss']
        }
        if self.alpha_dict == {}:
            out['loss'] = out['loss_driving'] + out['loss_classif'] 
        else:
            out['loss'] = sum(self.alpha_dict[k]*out[k] for k in self.alpha_dict)
        return out
