import os
import wandb
from src.utils import IS_DEBUG

import numpy as np

if IS_DEBUG:
    os.environ['WANDB_MODE'] = 'dryrun'


class WAndB:
    def __init__(self, name, model, hparams, type):
        assert type in ['video', 'image'], \
            '`type` should be "video" or "image"'

        self.type = type

        wandb.init(
            name=name,
            dir='.reports',
            project='arc',
            config=dict(
                vars(hparams),
                name=model.name,
                model_num_params=model.count_parameters(),
            ),
        )

        # ScriptModule models can't be watched for the current version of wandb
        try:
            wandb.watch(model)
        except Exception as _e:
            pass

        # wandb.save('data.py')
        # wandb.save('utils.py')
        # wandb.save('run_experiment.py')
        # [
        #     wandb.save(f'./models/{f}') for f in os.listdir('./models')
        #     if not f.startswith('__')
        # ]
        self.model = model

    def log(self, dict):
        wandb.log(dict)

    def log_images(self, name, imgs):
        wandb.log({name: [wandb.Image(i) for i in imgs]})

    def log_info(self, info, prefix='train'):
        import src.vis as vis

        if hasattr(self.model, 'scheduler'):
            wandb.log({'lr_scheduler': self.model.scheduler.get_lr()[0]})

        num_log_batches = 1
        y = info['y'][:num_log_batches].detach().cpu().numpy()
        y_pred = info['y_pred'][:num_log_batches].detach().cpu().numpy()
        diff = abs(y - y_pred)

        wandb.log({
            f'{prefix}_y': vis.plot_pictures(y[0]),
            f'{prefix}_y_pred': vis.plot_pictures(y_pred[0]),
            f'{prefix}_diff': vis.plot_pictures(diff[0])
        })
