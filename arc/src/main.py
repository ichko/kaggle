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
import src.data.preprocess as preprocess
import src.data.postprocess as postprocess

import matplotlib.pyplot as plt
import torch
import numpy as np
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

    score, solved = metrics.arc_eval(model, dataloader, hparams.nca_iterations)
    with torch.no_grad():
        epoch_info = model.optim_epoch(
            dataloader,
            preprocess=preprocess.strict_predict_all_tiles,
            postprocess=postprocess.standard,
            verbose=True,
        )

    infos = epoch_info['infos']
    loss_mean = epoch_info['loss_mean']
    loss_sort_index = epoch_info['loss_sort_index']
    names = infos['name'][loss_sort_index.tolist()]

    idx = 0
    logger.log_info(
        caption=infos['name'][idx],
        info=infos,
        prefix=prefix,
        idx=idx,
    )

    for desc in \
        ['test_len', 'test_in', 'test_out', 'test_pred_seq', 'test_pred']:
        infos[desc] = infos[desc][loss_sort_index]

    for desc, idx in zip(['best', 'middle', 'worst'],
                         [0, len(infos['test_len']) // 2, -1]):
        name = names[idx]
        logger.log_info(
            caption=name,
            info=infos,
            prefix=prefix + '_' + desc,
            idx=idx,
        )

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

    test_dl = data.load_arc_data(
        path='.data/test',
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
            batch = preprocess.stochastic_train(
                batch,
                num_train_samples=hparams.num_train_samples,
                num_test_samples=hparams.num_test_samples,
            )
            info = model.optim_step(batch)
            loss = info['loss']

            tq_batches.set_description(f'Loss: {loss:.6f}')
            logger.log({'train_loss': loss})

        if epoch % hparams.eval_interval == 0:
            log(model, train_dl, prefix='train', hparams=hparams)
            # log(model, val_dl, prefix='val', hparams=hparams)

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
