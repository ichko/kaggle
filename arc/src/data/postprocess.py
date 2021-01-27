CHANNEL_DIM = 2


def standard(batch, info):
    X, y = batch

    return {
        'name': X['name'],
        'loss': info['loss'],
        'test_len': X['test_len'],
        'test_in': X['test_in'].argmax(dim=CHANNEL_DIM),
        'test_out': y,
        'test_pred': info['test_pred'],
        'test_pred_seq': info['test_pred_seq'],
        'batch_losses': info['batch_losses'],
    }
