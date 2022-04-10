

from mebe.data import SequencesDataModule
from mebe.model import TransformerDenoisingModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # monitor="val_loss",
    monitor="val_loss",
    dirpath=".checkpoints/",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    model = TransformerDenoisingModel()
    dm = SequencesDataModule(bs=1)

    tb_logger = pl.loggers.TensorBoardLogger(".logs/")
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=200, verbose=False, mode="min")

    trainer = pl.Trainer(gpus=1, logger=tb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(
        model=model,
        train_dataloaders=dm.test_dataloader(),
        val_dataloaders=dm.train_dataloader(),
    )
