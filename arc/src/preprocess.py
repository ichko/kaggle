import torch
import src.nn_utils as ut

# TODO: Mask train pairs after using them for inference
CHANNEL_DIM = 2
NUM_CLASSES = 11


def strict(batch):
    X, _ = batch
    train_in = ut.one_hot(X['train_in'], NUM_CLASSES, CHANNEL_DIM)
    train_out = ut.one_hot(X['train_out'], NUM_CLASSES, CHANNEL_DIM)
    test_in = ut.one_hot(X['test_in'], NUM_CLASSES, CHANNEL_DIM)
    test_out = X['test_out'].long()

    train = torch.cat([train_in, train_out], dim=CHANNEL_DIM)

    return {
        'name': X['name'],
        'train_len': X['train_len'],
        'test_len': X['test_len'],
        'train': train,
        'test_in': test_in,
        'test_out': test_out,
    }, test_out


def strict_predict_all_tiles(batch):
    X, _ = batch
    train_in = ut.one_hot(X['train_in'], NUM_CLASSES, CHANNEL_DIM)
    train_out = ut.one_hot(X['train_out'], NUM_CLASSES, CHANNEL_DIM)
    train = torch.cat([train_in, train_out], dim=CHANNEL_DIM)

    all_in = ut.one_hot(X['in'], NUM_CLASSES, CHANNEL_DIM)
    all_out = X['out'].long()

    return {
        'name': X['name'],
        'train_len': X['train_len'],
        'test_len': X['len'],
        'train': train,
        'test_in': all_in,
        'test_out': all_out,
    }, all_out


def stochastic(batch, max_train=3, max_test=2):
    X, _ = batch
    lens = X['len']
    all_in = ut.one_hot(X['in'], NUM_CLASSES, CHANNEL_DIM)
    all_out = ut.one_hot(X['out'], NUM_CLASSES, CHANNEL_DIM)
    pairs = torch.cat([all_in, all_out], dim=CHANNEL_DIM)

    train, train_len = ut.sample_padded_sequences(pairs, lens, max_train)
    test, test_len = ut.sample_padded_sequences(pairs, lens, max_test)

    test_in, test_out = test.chunk(2, dim=CHANNEL_DIM)
    test_out = torch.argmax(test_out, dim=CHANNEL_DIM)

    return {
        'name': None,
        'train_len': train_len,
        'test_len': test_len,
        'train': train,
        'test_in': test_in,
        'test_out': test_out,
    }, test_out
