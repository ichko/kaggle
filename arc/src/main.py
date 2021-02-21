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
import src.nn_utils as ut

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

    return model.to(DEVICE)


def main(hparams):
    pp = pprint.PrettyPrinter(4)
    pp.pprint(vars(hparams))

    dataloaders = data.load(hparams=hparams, DEVICE=DEVICE)
    model = get_model(hparams)

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
        training_cycle = ut.make_cycle(
            model=model,
            dl=dataloaders['train'],
            preprocess=lambda batch: data.preprocess.stochastic_train(
                batch,
                num_train_samples=hparams.num_train_samples,
                num_test_samples=hparams.num_test_samples,
            ),
            postprocess=data.postprocess.standard,
        )
        tq_batches = tqdm(training_cycle)

        for idx, info in enumerate(tq_batches):
            loss = info['loss'].item()

            tq_batches.set_description(f'Loss: {loss:.6f}')
            logger.log({'train_loss': loss})

        if epoch % hparams.eval_interval == 0:
            model.eval()

            with torch.no_grad():
                eval_metric = metrics.ArcEval()
                log_cycle = ut.make_cycle(
                    model=model,
                    dl=dataloaders['train'],
                    preprocess=data.preprocess.strict_predict_all_tiles,
                    postprocess=data.postprocess.standard,
                )
                for info in tqdm(log_cycle):
                    # TODO: The design of this should be considered
                    # Take only the last demonstration pair as it is the actual test pair
                    # The test sequences are padded so the selection of the last
                    #   test input/output pair must be done using the lengths
                    #   of the sequences as index.

                    y_hat_batch = info['y_pred'][:, info['test_len'] - 1]
                    y_batch = info['y'][:, info['test_len'] - 1]

                    eval_metric.push(y_hat_batch, y_batch)

            score, solved = eval_metric.reduce()
            info = log_cycle.artefacts()

            infos = info['infos']
            loss_mean = info['loss_mean']
            loss_sort_index = info['loss_sort_index'].tolist()
            names = infos['name'][loss_sort_index]

            prefix = 'train'
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

            best_middle_worst = zip(['best', 'middle', 'worst'],
                                    [0, len(infos['test_len']) // 2, -1])
            best_worst = zip(['best', 'worst'], [0, -1])
            for desc, idx in best_worst:
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

            print('\n')
            print(f'======= LOG {prefix.upper()} =======')
            print(f'{prefix.upper()} LOSS  : {loss_mean:.6f}')
            print(f'{prefix.upper()} SOLVED: {solved}')
            print('\n')

            model.train()

            # model.persist()

        # model.save()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='id of configuration')
    parser.add_argument('--from-scratch', action='store_false')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    hparams = config.get_hparams(args.config)
    main(hparams)
