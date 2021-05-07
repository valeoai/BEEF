from pathlib import Path

from bootstrap.lib.options import Options
from .hdd import HDD
from .hdd_classif import HDDClassif
from .bdd import BDDDrive
from .bdd_caption import BDDCaption

def factory(engine=None):
    opt = Options()['dataset']
    dataset = {}
    if opt.get('train_split', None):
        dataset['train'] = factory_split(opt['train_split'], mode='train')
    if opt.get('eval_split', None):
        dataset['eval'] = factory_split(opt['eval_split'], mode="eval")

    return dataset

def factory_split(split, mode=None):
    opt = Options()['dataset']
    shuffle = mode == 'train'

    if opt['name'] == 'hdd_classif':
        dataset = HDDClassif(
            dir_data=Path(opt['dir_data']),
            split=split,
            win_size=opt['win_size'],
            im_size=opt.get('im_size', 'small'),
            layer=opt['layer'],
            frame_position=opt['frame_position'],
            traintest_mode=opt.get('traintest_mode', False),
            fps=opt['fps'],
            horizon=opt['horizon'],
            batch_size=opt['batch_size'],
            debug=opt['debug'],
            shuffle=shuffle,
            pin_memory=Options()['misc']['cuda'],
            nb_threads=opt['nb_threads']
        )
    elif opt['name'] == 'bdd_drive':
        dataset = BDDDrive(dir_data=Path(opt['dir_data']),
                           split=split,
                           n_before=opt['n_before'],
                           batch_size=opt['batch_size'],
                           debug=opt['debug'],
                           shuffle=shuffle,
                           pin_memory=Options()['misc']['cuda'],
                           nb_threads=opt['nb_threads'])
    elif opt['name'] == 'bdd_caption':
        dataset = BDDCaption(dir_data=Path(opt['dir_data']),
                           split=split,
                           n_before=opt['n_before'],
                           batch_size=opt['batch_size'],
                           features_dir=opt['features_dir'],
                           debug=opt['debug'],
                           shuffle=shuffle,
                           pin_memory=Options()['misc']['cuda'],
                           nb_threads=opt['nb_threads'])
    else:
        raise ValueError(opt['name'])

    return dataset

