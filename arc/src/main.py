import argparse

import src.data as data
import src.vis as vis
import src.models as models
import src.loggers as loggers
import src.config as config
import src.utils as utils

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
def evaluate(model, dataloader, num_iters):
    error = 0
    length = 0
    num_solved = 0
    losses = []

    for batch in tqdm(dataloader):
        X, y_batch = batch
        # Currently outputting single prediction per test input
        with torch.no_grad():
            y_hat_batch = model(X, num_iters)
            loss, _ = model.optim_step(batch)

        losses.append(loss)

        for y, y_hat in zip(y_batch, y_hat_batch):
            length += 1
            assert y_hat.shape == y.shape, \
                "The shapes of y and y_pred should match!!!"

            if torch.all(y_hat.int() == y.int()).item():
                num_solved += 1
            else:
                error += 1

    losses = np.array(losses).mean()
    return error / length, num_solved, losses


def get_model(hparams):
    import importlib

    model_module = importlib.import_module(f'src.models.{hparams.model}')
    model_module.sanity_check()

    model = model_module.make_model(vars(hparams))
    model.make_persisted(f'.models/{model.name}.weights')

    return model


def main(hparams):
    import pprint
    import sys
    from tqdm import tqdm

    pp = pprint.PrettyPrinter(4)
    pp.pprint(vars(hparams))

    train_dl = data.load_data(
        '.data/training',
        bs=hparams.bs,
        shuffle=True,
        device=DEVICE,
    )

    val_dl = data.load_data(
        '.data/evaluation',
        bs=hparams.bs,
        shuffle=False,
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

    if not utils.IS_DEBUG:
        print('\n\nPress ENTER to continue')
        _ = input()
        print('...')

    logger = loggers.WAndB(
        name=hparams.model,
        model=model,
        hparams=hparams,
        type='image',
    )

    # Summary for the logger
    model.summary()
    # torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(hparams.epochs)):
        # for num_iters in tqdm(range(1, hparams.nca_iterations, 10)):
        # model.set_num_iters(num_iters)

        tq_batches = tqdm(train_dl)
        for idx, batch in enumerate(tq_batches):
            # with ef.scan(wait=i == 0):
            loss, info = model.optim_step(batch)

            tq_batches.set_description(f'Loss: {loss:.6f}')
            logger.log({'train_loss': loss})

        if epoch % hparams.eval_interval == 0:
            train_score, train_solved, train_loss = \
                evaluate(model, train_dl, hparams.nca_iterations)
            val_score, val_solved, val_loss = \
                evaluate(model, val_dl, hparams.nca_iterations)

            idx = 0
            length = info['test_len'][idx]
            inputs = info['test_inputs'][idx, :length]
            outputs = info['test_outputs'][idx, :length]
            preds = info['test_preds'][idx, :length]
            preds_seq = info['test_preds_seq'][idx, :length]

            vid_path = f'.videos/pred{str(epoch)}.mp4'
            vis.save_task_vid(
                path=vid_path,
                inputs=inputs,
                outputs=outputs,
                preds_seq=preds_seq,
            ),
            logger.log_video('task', vid_path)

            logger.log({
                'train_loss_mean': train_loss,
                'val_loss_mean': val_loss,
                'train_y': vis.plot_grid(outputs[0]),
                'train_y_pred': vis.plot_grid(preds[0]),
                'train_score': train_score,
                'val_score': val_score,
                'train_solved': train_solved,
                'val_solved': val_solved,
            })

            print(f'====== EPOCH {epoch} END ======')
            print('FINAL TRAIN SCORE:', train_score)
            print('FINAL VAL SCORE:', val_score)

            # model.persist()

        # model.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='id of configuration',
    )
    parser.add_argument('--from-scratch', action='store_false')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    hparams = config.get_hparams(args.config)
    main(hparams)
