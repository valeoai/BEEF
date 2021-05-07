from bootstrap.lib.options import Options

from .bdd import BDDDrive
from .map import mAP
from .futuretraj import FutureTraj
from .bdd_caption import CaptionMetrics
from .multimetrics import MultiMetrics


def factory(engine, mode):
    opt = Options()['model.metric']
    if opt['name'] == 'map':
        metric = mAP(engine,
                     mode=mode)
    elif opt['name'] == 'future_traj':
        metric = FutureTraj(engine,
                            mode=mode)
    elif opt['name'] == 'multi_metrics':
        metric = MultiMetrics(engine, mode=mode, metrics=opt['metrics'])
    elif opt['name'] == "bdd-drive":
        metric = BDDDrive(engine, mode=mode)
    elif opt['name'] == 'bdd_caption':
        decode_fn = engine.dataset[mode].to_sentence
        bleu_smoothing = opt["bleu_smoothing"]
        output_sentence = Options().get("model.network.output_sentence", "caption")
        metric = CaptionMetrics(decode_fn, bleu_smoothing, engine, mode=mode, output_sentence=output_sentence)
    else:
        raise ValueError(opt['name'])
    return metric

