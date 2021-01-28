from src.data.load import load_arc_data
import src.data.preprocess as preprocess


class MapDataloader:
    def __init__(self, mapper, dataloader):
        self.mapper = mapper
        self.dataloader = dataloader

    def __iter__(self):
        self.it = iter(self.dataloader)
        return self

    def __next__(self):
        item = next(self.it)
        item = self.mapper(item)

        return item


def load(hparams, DEVICE):
    train_dl = load_arc_data(
        path='.data/training',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    val_dl = load_arc_data(
        path='.data/evaluation',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    test_dl = load_arc_data(
        path='.data/test',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    on_train = MapDataloader(preprocess.stochastic_train, train_dl)
    on_log = MapDataloader(preprocess.strict_predict_all_tiles, train_dl)

    return on_train, on_log
