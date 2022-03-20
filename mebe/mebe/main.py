import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_real_flies, get_fly_dists


def download_data():
    aicrowd_challenge_name = "mabe-2022-fruit-fly-groups"
    if not os.path.exists('data'):
        os.mkdir('data')

    # %aicrowd ds dl -c {aicrowd_challenge_name} -o data # Download all files
    # %aicrowd ds dl -c {aicrowd_challenge_name} -o data *submission_data* # download only the submission keypoint data
    # download only the submission keypoint data
    os.system(
        f'aicrowd ds dl -c {aicrowd_challenge_name} -o data *user_train*')

    os.system(
        f'aicrowd ds dl -c {aicrowd_challenge_name} -o data *submission_data* # download only the submission keypoint data')


if __name__ == '__main__':
    # download_data()

    user_train = np.load('data/user_train.npy', allow_pickle=True).item()

    # sample_submission = np.load('data/sample_submission.npy',allow_pickle=True).item()

    print("Dataset keys - ", user_train.keys())
    print("Number of train data sequences - ", len(user_train['sequences']))

    sequence_names = list(user_train["sequences"].keys())
    sequence_key = sequence_names[0]
    single_sequence = user_train["sequences"][sequence_key]
    print("Sequence name - ", sequence_key)
    print("Single Sequence shape ", single_sequence['keypoints'].shape)
    print(f"Number of elements in {sequence_key} - ", len(single_sequence))

    def seed_everything(seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    seed = 42
    seed_everything(seed)

    # parameters for embedding and sampling
    ntgtsclose = 2  # select this fly and the next closest fly
    nframesclose = 2  # select this and the next frame
    nsampleswant = 10000

    # sizes of various things
    nseq = len(user_train['sequences'])
    seqid = next(iter(user_train['sequences']))
    seq = user_train['sequences'][seqid]['keypoints']
    nframes = seq.shape[0]
    ntgts = seq.shape[1]
    nfeatures = seq.shape[2]
    ndim = seq.shape[3]
    dimsperframe = ntgtsclose*nfeatures*ndim
    # how many features will we use for each frame and fly
    highdim = dimsperframe * nframesclose
    embed_size = 256//ntgts  # number of dimensions per target and frame
    seq.shape

    # randomly choose a subset of the data
    nsamplestotal = 0
    for seqid, v in user_train['sequences'].items():
        seq = v['keypoints']

        # there may be < nflies flies in this sequence, ignore missing flies
        isreal = np.any(get_real_flies(seq), axis=0)
        nsamplestotal = nsamplestotal + \
            np.count_nonzero(isreal)*(nframes-nframesclose+1)

    # probability to choose any frame, fly
    psample = nsampleswant / nsamplestotal

    # sample beforehand so we know how many samples we will actually get
    dosample = {}
    nsamples = 0
    for seqid, v in user_train['sequences'].items():
        seq = v['keypoints']

        # there may be < nflies flies in this sequence, ignore missing flies
        isreal = np.any(get_real_flies(seq), axis=0)
        dosample[seqid] = np.random.rand(
            nframes-nframesclose+1, np.count_nonzero(isreal)) <= psample
        nsamples += np.count_nonzero(dosample[seqid])
    print(
        f'Subsampling {nsamples} frames, flies from a total of {nsamplestotal}')

    # create a basic feature representation from a sequence
    def localcontext(seq, tgt, t, nframesclose, ntgtsclose):
        dimsperframe = ntgtsclose*nfeatures*ndim
        highdim = dimsperframe * nframesclose

        highx = np.zeros(highdim)
        x = seq[t, ...]
        # find the ntgtsclose-1 flies closest to this fly
        d = get_fly_dists(x, tgt=tgt)
        order = np.argsort(d)
        order = order[:ntgtsclose]

        # store data for nframesclose frames, startint at t
        for off in range(nframesclose):
            t1 = np.minimum(t+off, nframes-1)
            x = seq[t1, order, ...]
            highx[off*dimsperframe:(off+1) *
                  dimsperframe] = seq[t1, order, ...].flatten()
        return highx

    pca_train = np.zeros((nsamples, highdim))

    with tqdm(total=nseq) as pbar:
        samplei = 0
        for seqid, v in user_train['sequences'].items():
            seq = v['keypoints']

            # there may be < nflies flies in this sequence, ignore missing flies
            isreal = np.any(get_real_flies(seq), axis=0)
            seq = seq[:, isreal, ...]
            ntgtscurr = seq.shape[1]

            for tgt in range(ntgtscurr):
                for t in range(nframes-nframesclose+1):
                    if not dosample[seqid][t, tgt]:
                        continue

                    pca_train[samplei, :] = localcontext(
                        seq, tgt, t, nframesclose, ntgtsclose)
                    samplei += 1
            pbar.update()

    # z-score to normalize data
    scaler = StandardScaler()
    scaler.fit(pca_train)
    pca_train = scaler.transform(pca_train)
    pca_train[np.isnan(pca_train)] = 0  # fill nans with mean

    # pca
    pca = PCA(n_components=embed_size)
    pca.fit(pca_train)

    # function to project a sequence onto these pcs
    def pcaproject(seq, pca, scaler, nframesclose, ntgtsclose):
        nframes = seq.shape[0]
        ntgts = seq.shape[1]
        nfeatures = seq.shape[2]
        ndim = seq.shape[3]
        dimsperframe = ntgtsclose*nfeatures*ndim
        highdim = dimsperframe * nframesclose
        lowdim = pca.n_components * ntgts
        embedding = np.zeros((nframes, lowdim))
        for t in range(nframes):
            for tgt in range(ntgts):
                highx = localcontext(seq, tgt, t, nframesclose, ntgtsclose)
                highx = scaler.transform(highx.reshape(1, highdim))
                highx[np.isnan(highx)] = 0.
                embedding[t, tgt*pca.n_components:(tgt+1)
                          * pca.n_components] = pca.transform(highx)
        return embedding

    # apply this embedding to a training sequence
    seqid = next(iter(user_train['sequences']))

    seq = user_train['sequences'][seqid]['keypoints']
    embedding = pcaproject(seq, pca, scaler, nframesclose, ntgtsclose)

    # plot different flies different colors
    fig, ax = plt.subplots()
    for tgt in range(ntgts):
        ax.plot(embedding[:, embed_size*tgt],
                embedding[:, embed_size*tgt+1], '.')
    _ = ax.axis('equal')
    _ = ax.set_xlabel('PC 1')
    _ = ax.set_ylabel('PC 2')
