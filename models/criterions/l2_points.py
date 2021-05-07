import torch
import torch.nn as nn

class L2Points(nn.Module):
    def __init__(self):
        super(L2Points, self).__init__()

    def forward(self, net_out, batch):
        """
         The loss is computed using the previous, future and current points. 
         """
        next_xy = batch['r_next_xy'] # b x 6 x 2
        prev_xy = batch['r_prev_xy']
        zeros = torch.zeros_like(next_xy[:,:1,:])
        gt_xy = torch.cat([prev_xy, zeros, next_xy], 1)

        pred_xy = net_out['pred_xy']
        bsize = next_xy.size(0)
        diff = pred_xy - gt_xy

        # loss_values = self.loss(pred_next_xy, next_xy)
        # Some points should not be used to train the model as they have a huge L2 loss. They are errors in 
        # the GPS side of the data acquisition/sampling/preprocessing.

        loss_values_mse = torch.sqrt(torch.sum((diff)**2, -1))
        lt_than_thresh = loss_values_mse.mean(-1) < 100
        loss_values_mse = loss_values_mse[lt_than_thresh]
        
        loss_value = loss_values_mse.mean()

        out = {'loss_driving': loss_value}

        out['loss'] = out['loss_driving']
        return out
