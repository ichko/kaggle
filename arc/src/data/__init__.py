from src.data.load import load_arc_data
import src.data.preprocess as preprocess
import src.data.postprocess as postprocess


def load(hparams, DEVICE):
    train = load_arc_data(
        path='.data/training',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    val = load_arc_data(
        path='.data/evaluation',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    test = load_arc_data(
        path='.data/test',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    return {
        'train': train,
        'val': val,
        'test': test,
    }
