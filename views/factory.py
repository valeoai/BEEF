from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from .plotly import Plotly

def factory():
    opt = Options()['view']
    exp_dir = Options()['exp.dir']

    view_name = opt.get('name', 'plotly')
    items = opt.get('items', None)
    fname = opt.get('file_name', 'view.html')

    if view_name == 'plotly':
        view = Plotly(items, exp_dir, fname)
    else:
        view = None
    return view

