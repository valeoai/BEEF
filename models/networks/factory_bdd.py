from pathlib import Path

from bootstrap.lib.options import Options

from .bdd import BDDDrive


def factory(engine):
    opt = Options()['model']['network']

    if opt['name'] == 'bdd-drive':
        net = BDDDrive(use_input_signals=opt['use_input_signals'])
    else:
        raise ValueError(opt['name'])
    return net
