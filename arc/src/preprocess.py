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
    test_out = ut.one_hot(X['test_out'], NUM_CLASSES, CHANNEL_DIM)
    y = X['test_out'].long()

    train = torch.cat([train_in, train_out], dim=CHANNEL_DIM)

    return {
        'name': X['name'],
        'train_len': X['train_len'],
        'test_len': X['test_len'],
        'train': train,
        'test_in': test_in,
        'test_out': test_out,
    }, y


def strict_predict_all_tiles(batch):
    X, _ = batch
    train_in = ut.one_hot(X['train_in'], NUM_CLASSES, CHANNEL_DIM)
    train_out = ut.one_hot(X['train_out'], NUM_CLASSES, CHANNEL_DIM)
    train = torch.cat([train_in, train_out], dim=CHANNEL_DIM)

    all_in = ut.one_hot(X['in'], NUM_CLASSES, CHANNEL_DIM)
    all_out = ut.one_hot(X['out'], NUM_CLASSES, CHANNEL_DIM)
    y = X['out'].long()

    return {
        'name': X['name'],
        'train_len': X['train_len'],
        'test_len': X['len'],
        'train': train,
        'test_in': all_in,
        'test_out': all_out,
    }, y


def _stochastic(lens, input, output, max_train=3, max_test=2):
    input = ut.one_hot(input, NUM_CLASSES, CHANNEL_DIM)
    output = ut.one_hot(output, NUM_CLASSES, CHANNEL_DIM)
    pairs = torch.cat([input, output], dim=CHANNEL_DIM)

    train, train_len = ut.sample_padded_sequences(pairs, lens, max_train)
    test, test_len = ut.sample_padded_sequences(pairs, lens, max_test)

    test_in, test_out = test.chunk(2, dim=CHANNEL_DIM)
    y = torch.argmax(test_out, dim=CHANNEL_DIM)

    return {
        'name': None,
        'train_len': train_len,
        'test_len': test_len,
        'train': train,
        'test_in': test_in,
        'test_out': test_out,
    }, y


def stochastic_all(batch):
    X, _ = batch
    return _stochastic(X['len'], X['in'], X['out'])


# TODO: Use this preprocessor while training to see if unseen test tiles
# are solved during evaluation
def stochastic_train(batch, max_train=3, max_test=2):
    X, _ = batch
    return _stochastic(X['train_len'], X['train_in'], X['train_out'])
