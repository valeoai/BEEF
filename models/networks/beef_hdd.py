import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import yaml 

import torch
import torch.nn as nn
import torchvision

from .mlp import MLP
from . import video_cnn
from .factory_fusions import factory as factory_fusions


class BeefHDD(nn.Module):
    def __init__(self, blinkers_dim, layers_to_fuse, label_fusion_opt, gru_opt, n_future, detach_pred):
        super(BeefHDD, self).__init__()

        self.cnn3d = video_cnn.r2plus1d_18(pretrained=True)
        in_features = self.cnn3d.fc.in_features
        self.cnn3d.fc = nn.Identity()
        if blinkers_dim>0:
            self.blinkers_emb = nn.Embedding(3,blinkers_dim)
        else:
            self.blinkers_emb = None


        self.layers_to_fuse = layers_to_fuse
        self.label_fusion = factory_fusions(label_fusion_opt)
        gru_opt['input_size'] = in_features + blinkers_dim
        gru_opt['batch_first'] = True
        self.n_future = n_future # Number of positions to sample
        self.gru_out = nn.GRU(**gru_opt)
        self.lin_out = nn.Linear(gru_opt['hidden_size'], 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.detach_pred = detach_pred

    def forward(self, batch):
        frames = batch['frames'].transpose(1,2)
        vis_features = self.cnn3d(frames)
        if self.blinkers_emb is not None:
            blink_features = self.blinkers_emb(batch['blinkers']).squeeze(1)
            input_features = torch.cat([vis_features['output'], blink_features], -1)
        else:
            input_features = vis_features['output']

        input_features = input_features[:,None,:].expand(-1, self.n_future, -1 )
        out_gru, _ = self.gru_out(input_features)

        pred_xy = self.lin_out(out_gru)
        vis_features['prediction'] = pred_xy.view(pred_xy.size(0), -1)
        if self.detach_pred:
            vis_features['prediction'] = vis_features['prediction'].detach()

        label_fusion_inputs = []
        for k in self.layers_to_fuse:
            v = vis_features[k]
            if v.dim() == 5:
                v = self.avgpool(v)
                v = v.flatten(1)
            label_fusion_inputs.append(v)

        y_pred = self.label_fusion(label_fusion_inputs)

        return {'y_pred': y_pred, 'pred_xy': pred_xy}

class DriverHDD(nn.Module):
    def __init__(self, blinkers_dim, gru_opt, n_future):
        super(DriverHDD, self).__init__()
        self.cnn3d = video_cnn.r2plus1d_18(pretrained=True)
        visfeatures_dim = self.cnn3d.fc.in_features
        self.cnn3d.fc = nn.Identity()
        self.blinkers_emb = nn.Embedding(3,blinkers_dim)
        gru_opt['input_size'] = visfeatures_dim + blinkers_dim
        gru_opt['batch_first'] = True
        self.n_future = n_future # Number of positions to sample
        self.gru_out = nn.GRU(**gru_opt)
        self.lin_out = nn.Linear(gru_opt['hidden_size'], 2)


    def forward(self, batch):
        blink_features = self.blinkers_emb(batch['blinkers']).squeeze(1)
        frames = batch['frames'].transpose(1,2)
        vis_features = self.cnn3d(frames)
        input_features = torch.cat([vis_features['output'], blink_features], -1)
        input_features = input_features[:,None,:].expand(-1, self.n_future, -1 )
        out_gru, _ = self.gru_out(input_features)

        pred_xy = self.lin_out(out_gru)
        out = {'pred_xy': pred_xy}
        out.update(vis_features)
        return out


class BaselineMultitaskHDD(nn.Module):
    def __init__(self, n_classes, blinkers_dim, layer_to_extract, dim_features, gru_opt, n_future, mlp_opt):
        super(BaselineMultitaskHDD, self).__init__()

        self.cnn3d = video_cnn.r2plus1d_18(pretrained=True)
        in_features = self.cnn3d.fc.in_features
        self.cnn3d.fc = nn.Identity()
        self.layer_to_extract = layer_to_extract
        self.blinkers_emb = nn.Embedding(3,blinkers_dim)


        gru_opt['input_size'] = in_features + blinkers_dim
        gru_opt['batch_first'] = True
        self.n_future = n_future # Number of positions to sample
        self.gru_out = nn.GRU(**gru_opt)
        self.lin_out = nn.Linear(gru_opt['hidden_size'], 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))


        if mlp_opt is not None:
            mlp_opt['input_dim'] = dim_features
            mlp_opt['dimensions'].append(n_classes)
            self.label_lin = MLP(**mlp_opt)
            n_params_mlp = sum([p.numel() for p in self.label_lin.parameters()])
            print("N_params_mlp =", n_params_mlp)
        else:
            self.label_lin = nn.Linear(dim_features, n_classes)


    def forward(self, batch):
        frames = batch['frames'].transpose(1,2)
        vis_features = self.cnn3d(frames)
        blink_features = self.blinkers_emb(batch['blinkers']).squeeze(1)
        extracted = vis_features[self.layer_to_extract]
        if extracted.dim() == 5:
            extracted = self.avgpool(extracted)
            extracted = extracted.flatten(1)
        y_pred = self.label_lin(extracted)

        input_features = torch.cat([vis_features['output'], blink_features], -1)
        input_features = input_features[:,None,:].expand(-1, self.n_future, -1 )
        out_gru, _ = self.gru_out(input_features)

        pred_xy = self.lin_out(out_gru)
        return {'y_pred': y_pred, 'pred_xy': pred_xy}