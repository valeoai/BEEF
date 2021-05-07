import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from . import video_cnn

import torchvision
import torch
import torch.nn as nn

from .mlp import MLP


class BDDDrive(nn.Module):
    def __init__(self, use_input_signals):
        super(BDDDrive, self).__init__()
        self.use_input_signals = use_input_signals
        self.cnn3d = video_cnn.r2plus1d_18(pretrained=True)
        # self.cnn3d = torchvision.models.video.r2plus1d_18(pretrained=True)
        visfeatures_dim = self.cnn3d.fc.in_features
        self.cnn3d.fc = nn.Identity()
        if self.use_input_signals:
            input_dim = visfeatures_dim+2
        else:
            input_dim = visfeatures_dim
        self.lin_out = nn.Linear(input_dim, 2)

    def forward(self, batch):
        frames = batch['frames'].transpose(1,2)
        vis_features = self.cnn3d(frames)
        if self.use_input_signals:
            features = torch.cat([vis_features['output'], batch['goaldir_value'], batch['speed_value']], -1)
        else:
            features = vis_features['output']
        preds = self.lin_out(features)
        out = {'preds':preds}
        out.update(vis_features)
        return out
