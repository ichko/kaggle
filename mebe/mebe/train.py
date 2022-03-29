

from mebe.data import MaskedSequencesDataModule
from mebe.model import TransformerDenoisingModel
import pytorch_lightning as pl

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # monitor="val_loss",
    monitor="train_loss",
    dirpath=".checkpoints/",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    model = TransformerDenoisingModel()
    dm = MaskedSequencesDataModule(bs=1)

    tb_logger = pl.loggers.TensorBoardLogger(".logs/")

    trainer = pl.Trainer(gpus=1, logger=tb_logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(
        model=model,
        train_dataloader=dm.train_dataloader()
    )
