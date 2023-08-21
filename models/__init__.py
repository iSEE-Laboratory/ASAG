
from .asag import build
from .lr_scheduler import build_lr_scheduler

def build_model(args):
    return build(args)
