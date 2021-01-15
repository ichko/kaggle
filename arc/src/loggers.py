import os
import wandb
from src.utils import IS_DEBUG
import src.vis as vis

import numpy as np

if IS_DEBUG:
    os.environ['WANDB_MODE'] = 'dryrun'


class WAndB:
    def __init__(self, name, model, hparams):
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

        self.model = model

    def log(self, dict):
        wandb.log(dict)

    def log_info(self, info, prefix, idx=0):
        length = info['test_len'][idx]
        inputs = info['test_inputs'][idx, :length]
        outputs = info['test_outputs'][idx, :length]
        preds = info['test_preds'][idx, :length]
        preds_seq = info['test_preds_seq'][idx, :length]

        vid_path = '.temp/last_pred_vid.mp4'
        vis.save_task_vid(
            path=vid_path,
            inputs=inputs,
            outputs=outputs,
            preds_seq=preds_seq,
        ),
        wandb.log({f'{prefix}_task': wandb.Video(vid_path)})

        wandb.log({
            f'{prefix}_y': vis.plot_grid(outputs[0]),
            f'{prefix}_y_pred': vis.plot_grid(preds[0]),
        })
