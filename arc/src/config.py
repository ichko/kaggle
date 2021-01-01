from argparse import Namespace

defaults = dict(
    eval_interval=2,
    lr=0.00001,
    epochs=200,
    model='0',
    bs=20,
)

configs = dict(
    hyper_recurrent_cnn=dict(
        model='hyper_recurrent_cnn',
        nca_iterations=100,
        input_channels=11,
    ),
    _=None,
)


def get_hparams(config_id):
    config = configs[config_id]
    config_dict = {**defaults, **config}

    return Namespace(**config_dict)
