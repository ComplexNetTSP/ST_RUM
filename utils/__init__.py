from .random_walk import uniform_random_walk, uniqueness
from .seed import seed_everything
from .custom_metric import MaskedRMSE
from .custom_callback import TimingCallback
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')