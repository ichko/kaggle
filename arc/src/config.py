import sys
from argparse import Namespace
from collections import OrderedDict

IS_DEBUG = '--debug' in sys.argv

defaults = {
    'eval_interval': 20,
    'lr': 0.00005,
    'epochs': 1_000_000,
    'bs': 20,
}

configs = [
    {
        'model': 'hyper_recurrent_cnn',
        'nca_iterations': 10,
        'input_channels': 11,
        'num_train_samples': 3,
        'num_test_samples': 2,
    },
]

default_id = configs[0]['model']


def get_hparams(config_id):
    if config_id == 'default':
        config_id = default_id

    [config] = [c for c in configs if c['model'] == config_id]

    config_dict = {**defaults, **config}

    return Namespace(**config_dict)
