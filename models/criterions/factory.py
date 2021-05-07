from bootstrap.lib.options import Options

from .bdd import BDDDriveLoss
from .l2_points import L2Points
from .multitask import MultiTaskHDD
from .bdd_caption import BDDCaptionLoss
from .frames_classif import FramesClassifLoss


def factory(engine, mode):
    opt = Options()['model.criterion']
    if opt['name'] == "multitask_hdd":
        if opt['use_class_weights']:
            class_freq = engine.dataset[mode].class_freq
        else:
            class_freq = None
        criterion = MultiTaskHDD(class_freq=class_freq,
                                 alpha_dict=opt.get("alpha", {}))
    elif opt['name'] == 'l2_points':
        criterion = L2Points()
    elif opt['name'] == "bdd-drive":
        criterion = BDDDriveLoss(scales=opt['scales'],
                                 normalize_outputs=opt.get('normalize_outputs', False))
    elif opt['name'] == "bdd_caption":
        output_sentence = Options().get('model.network.output_sentence', "caption")
        criterion = BDDCaptionLoss(output_sentence)
    else:
        raise ValueError(opt['name'])

    return criterion
