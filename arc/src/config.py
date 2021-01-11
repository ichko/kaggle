from argparse import Namespace

defaults = dict(
    eval_interval=5,
    lr=0.00005,
    epochs=1_000_000,
    bs=20,
)

# TODO: Infer in latent space
configs = dict(
    hyper_recurrent_cnn=dict(
        model='hyper_recurrent_cnn',
        nca_iterations=20,
        input_channels=11,
        latent_space_inference=False,
    ),
    _=None,
)


def get_hparams(config_id):
    config = configs[config_id]
    config_dict = {**defaults, **config}

    return Namespace(**config_dict)
