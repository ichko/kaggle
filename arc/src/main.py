import src.data as data
import src.vis as vis
import src.models as models
import src.loggers as loggers
import src.config as config
import src.utils as utils

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
def evaluate(model, dataloader):
    error = 0
    for X, y in tqdm(dataloader):
        # Currently outputting single prediction per test input
        y_hat = model(X)

        assert y_hat.shape == y.shape, \
            "The shapes of y and y_pred should match!!!"

        task_error = 1
        if torch.all(y_hat == y).item():
            task_error = 0

        error += task_error

    return error / len(dataloader)


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

    print('DEVICE:', DEVICE)
    print(f'## Start training with configuration "{hparams.model.upper()}"')

    if not utils.IS_DEBUG:
        print('\n\nPress ENTER to continue')
        _ = input()
        print('...')

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
        bs=5,
        shuffle=False,
        device=DEVICE,
    )

    model = get_model(hparams)
    model = model.to(DEVICE)

    logger = loggers.WAndB(
        name=hparams.model,
        model=model,
        hparams=hparams,
        type='image',
    )

    if 'from_scratch' in sys.argv:
        try:
            model.preload_weights()
            print('>>> MODEL PRELOADED')
        except Exception as e:
            raise Exception(f'>>> Could not preload! {str(e)}')

    model.configure_optim(lr=hparams.lr)
    model.summary()

    for epoch in tqdm(range(hparams.epochs)):
        # for num_iters in tqdm(range(1, hparams.nca_iterations, 10)):
        # model.set_num_iters(num_iters)

        tq_batches = tqdm(train_dl)
        for idx, batch in enumerate(tq_batches):
            loss, info = model.optim_step(batch)

            tq_batches.set_description(f'Loss: {loss:.6f}')
            logger.log({'train_loss': loss})

        if epoch % hparams.eval_interval == 0:
            train_score = evaluate(model, train_dl)
            val_score = evaluate(model, val_dl)

            idx = 0
            length = info['test_len'][idx]
            inputs = info['test_inputs'][idx, :length]
            outputs = info['test_outputs'][idx, :length]
            preds = info['test_preds'][idx, :length]

            logger.log({
                'task':
                vis.plot_task_inference(
                    inputs=inputs,
                    outputs=outputs,
                    preds=preds,
                ),
                'train_y':
                vis.plot_grid(outputs[0]),
                'train_y_pred':
                vis.plot_grid(preds[0]),
                'train_score':
                train_score,
                'val_score':
                val_score,
            })

            print(f'====== EPOCH {epoch} END ======')
            print('FINAL TRAIN SCORE:', train_score)
            print('FINAL VAL SCORE:', val_score)

            model.persist()

        model.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='id of configuration',
    )
    # parser.add_argument('--from-scratch', action='store_false')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    hparams = config.get_hparams(args.config)
    main(hparams)
