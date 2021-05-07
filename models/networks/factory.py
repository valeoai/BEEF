from pathlib import Path

import torchvision
import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel

from .beef_hdd import BaselineMultitaskHDD, BeefHDD, DriverHDD


def factory(engine):
    opt = Options()['model.network']
    if opt['name'] == "beef_hdd":
        net = BeefHDD(layers_to_fuse=opt['layers_to_fuse'],
                     label_fusion_opt=opt['label_fusion'],
                     blinkers_dim=opt['blinkers_dim'],
                     gru_opt=opt['gru_opt'],
                     n_future=opt['n_future'],
                     detach_pred=opt.get('detach_pred',False))
    elif opt['name'] == "driver_hdd":
        net = DriverHDD(blinkers_dim=opt['blinkers_dim'],
                                 gru_opt=opt['gru_opt'],
                                 n_future=opt['n_future'])
    elif opt['name'] == "baseline_multitask_hdd":
        net = BaselineMultitaskHDD(n_classes=opt['n_classes'],
                                    blinkers_dim=opt['blinkers_dim'],
                                    layer_to_extract=opt['layer_to_extract'],
                                    dim_features=opt['dim_features'],
                                    gru_opt=opt['gru_opt'],
                                    n_future=opt['n_future'],
                                    mlp_opt=opt.get('mlp_opt',None))
    else:
        raise ValueError(opt['name'])
    if torch.cuda.device_count()>1:
        net = DataParallel(net)
    return net
