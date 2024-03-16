import datetime
import numpy as np


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
    
def args_to_str(args):
    """Convert cmd line args into a logdir string for experiment logging"""
    exp_name = '{}/'.format(args.expname)
    exp_name += '-{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    return exp_name