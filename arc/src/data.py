import os
import json
import numpy as np
import src.utils as utils

import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence


def matrix_to_np(obj):
    obj = obj.copy()

    iterator = obj.items() if type(obj) == dict else enumerate(obj)

    for k, v in iterator:
        is_matrix = type(v) == list and type(v[0]) == list
        if is_matrix:
            obj[k] = np.array(v).astype(np.uint8)
        else:
            obj[k] = matrix_to_np(v)

    return obj


def load_folder(path):
    result = dict()

    file_names = os.listdir(path)
    file_names = file_names[:8]

    if utils.IS_DEBUG:
        # load only 10 files in debug mode
        file_names = file_names[:32]

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            json_content = json.loads(text)
            json_content = matrix_to_np(json_content)

            result[file_name[:-5]] = json_content

    return result


def one_hot_channels(inp, max_size):
    img = np.full(
        (inp.shape[0], max_size, inp.shape[1], inp.shape[2]),
        0,
        dtype=np.uint8,
    )

    for i in range(max_size):
        img[:, i] = inp == i

    return img


def normalize_matrix_size(matrix):
    BORDER_IDENTIFIER = 10

    matrix = np.pad(
        matrix,
        ((1, 1), (1, 1)),
        'constant',
        constant_values=(
            (BORDER_IDENTIFIER, BORDER_IDENTIFIER),
            (BORDER_IDENTIFIER, BORDER_IDENTIFIER),
        ),
    )
    max_width = 32
    max_height = 32

    h, w = matrix.shape

    padding_x = (max_width - w) // 2
    padding_y = (max_height - h) // 2

    output = np.zeros((max_width, max_height))
    output[padding_y:padding_y + h, padding_x:padding_x + w] = matrix

    return output


def normalize_obj(obj):
    def map_np(mapper, obj):
        if type(obj) is np.ndarray:
            return mapper(obj)

        iterator = obj.items() \
            if type(obj) is dict \
            else enumerate(obj)

        for k, v in iterator:
            obj[k] = map_np(mapper, v)

        return obj

    return map_np(normalize_matrix_size, obj)


def pad_in_dim(tensor, pad_size, dim, val=0):
    shape = list(tensor.shape)
    shape[dim] = pad_size - shape[dim]
    padding = torch.full(shape, val).to(tensor.device)
    out = torch.cat([tensor, padding], dim=dim)

    return out


def load_data(path, bs, shuffle, device='cpu'):
    data = load_folder(path)
    data = normalize_obj(data)
    tasks = list(data.values())

    max_train_pairs = 4  # demonstrations are padded to this
    max_test_pairs = 1
    num_colors = 11  # each color + one id for border
    seq_dim = 0

    def get(selectors, idx, max_pairs):
        train_test, io = selectors
        data = [t[io] for t in tasks[idx][train_test]]
        data = data[:max_pairs]
        data = np.array(data)
        # data = one_hot_channels(data, max_size=num_colors)
        # data = data.astype(np.float32)
        data_len = len(data)
        data = torch.Tensor(data)
        data = pad_in_dim(data, pad_size=max_pairs, dim=seq_dim)
        data = data.to(device)

        return data, data_len

    class Dataset(td.Dataset):
        def __len__(self):
            return len(tasks)

        def __getitem__(self, id):
            # TODO: Vary the paris which are used to do param inference
            train_in, train_len = get(['train', 'input'], id, max_train_pairs)
            train_out, _ = get(['train', 'output'], id, max_train_pairs)
            test_in, test_len = get(['test', 'input'], id, max_test_pairs)
            test_out, _ = get(['test', 'output'], id, max_test_pairs)

            # Infer over each demonstration, both in train and test pairs
            all_in = torch.cat([train_in[:train_len], test_in[:test_len]])
            all_out = torch.cat([train_out[:train_len], test_out[:test_len]])
            all_len = train_len + test_len
            max_all_pairs = max_train_pairs + max_test_pairs

            all_in = pad_in_dim(all_in, pad_size=max_all_pairs, dim=seq_dim)
            all_out = pad_in_dim(all_out, pad_size=max_all_pairs, dim=seq_dim)

            return dict(
                train_inputs=train_in,
                train_outputs=train_out,
                train_len=train_len,
                #
                test_inputs=test_in,
                test_outputs=test_out,
                test_len=test_len,
                #
                all_inputs=all_in,
                all_outputs=all_out,
                all_len=all_len,
            ), all_out,

    dl = td.DataLoader(
        Dataset(),
        batch_size=bs,
        shuffle=shuffle,
    )

    return dl
