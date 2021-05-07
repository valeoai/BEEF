import copy

from bootstrap.lib.logger import Logger

from .fusions import (Bilinear, Block, BlockTucker, ConcatMLP, LinearSum, MCB,
                      MFB, MFH, MLB, Mutan, Tucker)


def factory(opt):

    opt = copy.copy(opt)
    ftype = opt.pop('type', None) # rm type from dict

    if ftype == 'block':
        fusion = Block(**opt)
    elif ftype == 'block_tucker':
        fusion = BlockTucker(**opt)
    elif ftype == 'mlb':
        fusion = MLB(**opt)
    elif ftype == 'mfb':
        fusion = MFB(**opt)
    elif ftype == 'mfh':
        fusion = MFH(**opt)
    elif ftype == 'mcb':
        fusion = MCB(**opt)
    elif ftype == 'mutan':
        fusion = Mutan(**opt)
    elif ftype == 'tucker':
        fusion = Tucker(**opt)
    elif ftype == 'linear_sum':
        fusion = LinearSum(**opt)
    elif ftype == 'cat_mlp':
        fusion = ConcatMLP(**opt)
    elif ftype == 'bilinear':
        fusion = Bilinear(**opt)
    else:
        raise ValueError()

    Logger().log_value('nb_params_fusion', fusion.n_params, should_print=True)
    return fusion
