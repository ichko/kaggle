import os
import wandb

import src.utils as utils
import src.vis as vis

if utils.IS_DEBUG:
    os.environ['WANDB_MODE'] = 'dryrun'


def init(name, model, hparams):
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


def log(dict):
    wandb.log(dict)


def log_info_to_local_dir(info, vid_path, idx=0):
    length = info['test_len'][idx]
    inputs = info['test_in'][idx, :length]
    outputs = info['test_out'][idx, :length]
    preds_seq = info['test_pred_seq'][idx, :length]

    vis.save_task_vid(
        path=vid_path,
        inputs=inputs,
        outputs=outputs,
        preds_seq=preds_seq,
    )


def log_info(info, prefix, idx=0):
    vid_path = '.temp/last_pred_vid.mp4'
    log_info_to_local_dir(info, vid_path, idx)

    length = info['test_len'][idx]
    outputs = info['test_out'][idx, :length]
    preds = info['test_pred'][idx, :length]

    wandb.log({f'{prefix}_task': wandb.Video(vid_path)})

    wandb.log({
        f'{prefix}_y': vis.plot_grid(outputs[0]),
        f'{prefix}_y_pred': vis.plot_grid(preds[0]),
    })
