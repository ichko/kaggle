from argparse import Namespace

defaults = dict(
    eval_interval=10,
    lr=0.0001,
    epochs=100,
    model='0',
)

configs = dict(
    soft_compute_nca=dict(
        model='soft_compute_nca',
        nca_iterations=10,
        input_channels=11,
    ),
    _=None,
)


def get_hparams(config_id):
    config = configs[config_id]
    config_dict = {**defaults, **config}

    return Namespace(**config_dict)