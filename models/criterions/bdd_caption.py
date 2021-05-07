import torch
import torch.nn as nn


class BDDCaptionLoss(nn.Module):
    def __init__(self, output_sentence="caption"):
        super(BDDCaptionLoss, self).__init__()
        self.loss = nn.NLLLoss()
        self.output_sentence = output_sentence

    def forward(self, net_out, batch):
        y_pred = net_out["predicted_sentence_proba"].transpose(1, 2) # Put the vocab_size in dimension 1
        y_true = batch[self.output_sentence][:, 1:] # Remove the first token
        loss_value = self.loss(y_pred, y_true)
        return {'loss': loss_value}

