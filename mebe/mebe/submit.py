from mebe.data import SequencesDataModule, SequencesDataset
from mebe.model import TransformerDenoisingModel
import numpy as np
from tqdm.auto import tqdm


def validate_submission(submission, submission_clips):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if 'frame_number_map' not in submission:
        print("Frame number map missing")
        return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 256:
        print("Embeddings too large, max allowed is 256")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end-start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False

    if not len(submission['embeddings']) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission['embeddings']).all():
        print(f"Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True


if __name__ == "__main__":
    checkpoint_path = './.checkpoints/model-epoch=80-val_loss=0.14.ckpt'
    DEVICE = 'cuda'

    # model = TransformerDenoisingModel()
    model = TransformerDenoisingModel.load_from_checkpoint(checkpoint_path)

    model = model.to(DEVICE)
    model = model.eval()

    dm = SequencesDataModule(bs=1)
    test_dl = dm.test_dataloader()

    names, seqs = next(iter(test_dl))
    nframes = seqs.shape[1]
    num_total_frames = nframes * len(test_dl)
    embeddings_array = np.empty(
        (num_total_frames, model.embed_size), dtype=np.float32)
    frame_number_map = {}
    start = 0

    for names, seqs in tqdm(test_dl):
        seqs = seqs.to(DEVICE)
        embeddings = model.embed(seqs)

        for i in range(len(names)):
            name = names[i]
            seq = seqs[i]

            end = start + nframes
            np_embeddings = embeddings[i].detach().cpu().numpy()
            embeddings_array[start:end, :] = np_embeddings
            frame_number_map[name] = (start, end)
            start = end

    assert end == num_total_frames
    submission_dict = {"frame_number_map": frame_number_map,
                       "embeddings": embeddings_array}

    validate_submission(
        submission_dict, SequencesDataset.TEST)

    np.save("submission.ignore.npy", submission_dict)

    # aicrowd_challenge_name = "mabe-2022-fruit-fly-groups"
    # %aicrowd submission create --description "TransformerDenoisingModel" -c {aicrowd_challenge_name} -f submission.npy
