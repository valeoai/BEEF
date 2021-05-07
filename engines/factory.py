from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from .extract_engine import ExtractEngine
from .predict_engine import PredictEngine

def factory():
    if Options()['engine']['name'] == 'extract':
        engine = ExtractEngine()
    elif Options()['engine']['name'] == 'predict':
        opt = Options()['engine']
        engine = PredictEngine(vid_id=opt.get('vid_id', None))
    else:
        raise ValueError
    return engine