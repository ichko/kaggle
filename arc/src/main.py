import os
import sys
import pprint
from datetime import datetime

import src.data as data
import src.vis as vis
import src.models as models
import src.logger as logger
import src.config as config
import src.utils as utils
import src.metrics as metrics
import src.preprocess as preprocess

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

RUN_ID = f'run_{datetime.now()}'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(hparams):
    import importlib

    model_module = importlib.import_module(f'src.models.{hparams.model}')
    model_module.sanity_check()

    model = model_module.make_model(vars(hparams))
    model.make_persisted(f'.models/{model.name}_{RUN_ID}.weights')

    return model


def log(model, dataloader, prefix, hparams):
    model.eval()

    batch = next(iter(dataloader))
    batch = preprocess.strict_predict_all_tiles(batch)

    # Log example task
    with torch.no_grad():
        _loss, info = model.optim_step(batch)

    idx = 0
    X, _y = batch
    name = X['name'][idx]
    logger.log_info(caption=name, info=info, prefix=prefix, idx=idx)

    score, solved = metrics.arc_eval(model, dataloader, hparams.nca_iterations)

    losses = []
    for batch in tqdm(dataloader):
        batch = preprocess.strict(batch)
        with torch.no_grad():
            loss, info = model.optim_step(batch)
            losses.append(loss)

    loss_mean = torch.Tensor(losses).mean()

    logger.log({
        f'{prefix}_loss_mean': loss_mean,
        f'{prefix}_score': score,
        f'{prefix}_solved': solved,
    })

    print('\n', flush=True)
    print(f'======= LOG {prefix.upper()} =======', flush=True)
    print(f'{prefix.upper()} LOSS  :', loss_mean, flush=True)
    print(f'{prefix.upper()} SOLVED:', solved, flush=True)
    print('\n', flush=True)

    model.train()


def main(hparams):
    pp = pprint.PrettyPrinter(4)
    pp.pprint(vars(hparams))

    train_dl = data.load_arc_data(
        path='.data/training',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    val_dl = data.load_arc_data(
        path='.data/evaluation',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    model = get_model(hparams)
    model = model.to(DEVICE)

    if '--from-scratch' not in sys.argv:
        try:
            model.preload_weights()
            print('>>> MODEL PRELOADED')
        except Exception as e:
            raise Exception(f'>>> Could not preload! {str(e)}')
    else:
        print('>>> TRAINING MODEL FROM SCRATCH!')

    model.configure_optim(lr=hparams.lr)
    model.summary()

    print('DEVICE:', DEVICE)
    print(f'## Start training with configuration "{hparams.model.upper()}"')

    if not config.IS_DEBUG:
        print('\n\nPress ENTER to continue')
        _ = input()
        print('...')

    logger.init(
        name=hparams.model,
        model=model,
        hparams=hparams,
    )

    # Summary for the logger
    model.summary()

    for epoch in tqdm(range(hparams.epochs)):
        # for num_iters in tqdm(range(1, hparams.nca_iterations, 10)):
        # model.set_num_iters(num_iters)

        tq_batches = tqdm(train_dl)
        for idx, batch in enumerate(tq_batches):
            batch = preprocess.stochastic_train(batch)
            loss, info = model.optim_step(batch)

            tq_batches.set_description(f'Loss: {loss:.6f}')
            logger.log({'train_loss': loss})

        if epoch % hparams.eval_interval == 0:
            log(model, train_dl, prefix='train', hparams=hparams)
            log(model, val_dl, prefix='val', hparams=hparams)

            model.persist()

        model.save()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='id of configuration')
    parser.add_argument('--from-scratch', action='store_false')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    hparams = config.get_hparams(args.config)
    main(hparams)
