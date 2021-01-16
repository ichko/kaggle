from argparse import Namespace

defaults = dict(
    eval_interval=20,
    lr=0.00005,
    epochs=1_000_000,
    bs=20,
)

configs = dict(
    hyper_recurrent_cnn=dict(
        model='hyper_recurrent_cnn',
        nca_iterations=10,
        input_channels=11,
    ),
    _=None,
)


def get_hparams(config_id):
    config = configs[config_id]
    config_dict = {**defaults, **config}

    return Namespace(**config_dict)
