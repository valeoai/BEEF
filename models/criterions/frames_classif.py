import torch
import torch.nn as nn

class FramesClassifLoss(nn.Module):
    def __init__(self,
                 class_freq=None):
        super(FramesClassifLoss, self).__init__()
        if class_freq is not None:
            class_weights = torch.zeros(len(class_freq)).cuda()
            for k in class_freq:
                class_weights[int(k)] = 1/class_freq[k]
        else:
            class_weights = None
        self.loss = nn.CrossEntropyLoss(class_weights)

    def forward(self, net_out, batch):
        y_pred = net_out['y_pred']
        if y_pred.dim() == 3:
            y_pred = net_out['y_pred'].transpose(1,2)
        y_true = batch['y_true']
        loss_value = self.loss(y_pred, y_true)
        return {'loss':loss_value}