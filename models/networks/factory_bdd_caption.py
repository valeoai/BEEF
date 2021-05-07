from pathlib import Path

from bootstrap.lib.options import Options

from .bdd_caption import Captioning


def factory(engine):
    opt = Options()['model']['network']

    if opt['name'] == 'bdd_caption':
        net = Captioning(hidden_size=opt["lstm_hidden_size"],
                temperature=opt["temperature"],
                sampling_strategy=opt["sampling_strategy"],
                fusion_opt=opt["fusion"],
                gru_lstm=opt["gru_lstm"],
                output_sentence=opt.get("output_sentence", "caption"),
                )
    else:
        raise ValueError(opt['name'])
    return net

