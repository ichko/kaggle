import os
import json
import numpy as np
import src.utils as utils

import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence

import src.nn_utils as ut


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
    # file_names = file_names[:8]

    if utils.IS_DEBUG:
        # limit files in debug mode
        file_names = file_names[:32]

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            json_content = json.loads(text)
            json_content = matrix_to_np(json_content)

            result[file_name[:-5]] = json_content

    return result


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


def load_arc_data(path, bs, shuffle, device='cpu'):
    tasks_dict = load_folder(path)
    tasks_dict = normalize_obj(tasks_dict)
    tasks = list(tasks_dict.items())

    max_train_pairs = 4  # demonstrations are padded to this
    max_test_pairs = 1
    seq_dim = 0

    def get(selectors, task, max_pairs):
        train_test, io = selectors
        data = torch.Tensor([t[io] for t in task[train_test]])
        data = ut.fix_dim_size(data, max_pairs, dim=seq_dim)
        data = data.to(device)

        return data, len(data)

    class Dataset(td.Dataset):
        def __len__(self):
            return len(tasks)

        def __getitem__(self, idx):
            name, task = tasks[idx]
            train_in, train_len = \
                get(['train', 'input'], task, max_train_pairs)
            train_out, _ = get(['train', 'output'], task, max_train_pairs)
            test_in, test_len = get(['test', 'input'], task, max_test_pairs)
            test_out, _ = get(['test', 'output'], task, max_test_pairs)

            all_in = torch.cat([train_in[:train_len], test_in[:test_len]])
            all_out = torch.cat([train_out[:train_len], test_out[:test_len]])
            all_len = train_len + test_len
            max_all_pairs = max_train_pairs + max_test_pairs

            all_in = ut.fix_dim_size(all_in, max_all_pairs, dim=seq_dim)
            all_out = ut.fix_dim_size(all_out, max_all_pairs, dim=seq_dim)

            return dict(
                name=name,
                #
                train_input=train_in,
                train_output=train_out,
                train_len=train_len,
                #
                test_input=test_in,
                test_output=test_out,
                test_len=test_len,
                #
                input=all_in,
                output=all_out,
                len=all_len,
            ), test_out,

    dl = td.DataLoader(
        Dataset(),
        batch_size=bs,
        shuffle=shuffle,
    )

    return dl
