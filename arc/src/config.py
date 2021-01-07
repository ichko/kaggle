from argparse import Namespace

defaults = dict(
    eval_interval=50,
    lr=0.0001,
    epochs=1_000_000,
    model='0',
    bs=8,
)

configs = dict(
    hyper_recurrent_cnn=dict(
        model='hyper_recurrent_cnn',
        nca_iterations=50,
        input_channels=11,
        latent_space_inference=False,
    ),
    _=None,
)


def get_hparams(config_id):
    config = configs[config_id]
    config_dict = {**defaults, **config}

    return Namespace(**config_dict)
