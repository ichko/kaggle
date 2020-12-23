import os
import json
import numpy as np
import utils


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
    if utils.IS_DEBUG:
        # load only 10 files in debug mode
        file_names = file_names[:10]

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            json_content = json.loads(text)
            json_content = matrix_to_np(json_content)

            result[file_name[:-5]] = json_content

    return result


def map_np(mapper, obj):
    if type(obj) is np.ndarray:
        return mapper(obj)

    iterator = obj.items() \
        if type(obj) is dict \
        else enumerate(obj)

    for k, v in iterator:
        obj[k] = map_np(mapper, v)

    return obj


BORDER_IDENTIFIER = 10


def normalize_matrix_size(matrix):
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


def one_hot_channels(inp, max_size):
    inp = np.array(inp)
    img = np.full((max_size, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(max_size):
        img[i] = inp == i
    return img


def normalize_obj(obj):
    return map_np(normalize_matrix_size, obj)


def get_tasks_dl(tasks, bs, shuffle):
    # Wont work with bs != 1 because different task have
    # different number of examples
    assert bs == 1, "bs, wont work for values != 1"

    import torch.utils.data as td

    class Dataset(td.Dataset):
        def __len__(self):
            return len(tasks)

        def __getitem__(self, idx):
            num_semantic_ids = 11  # each color + one id for border
            train_inputs = np.array([
                one_hot_channels(t['input'], max_size=num_semantic_ids)
                for t in tasks[idx]['train']
            ], )
            train_outputs = np.array([
                one_hot_channels(t['output'], max_size=num_semantic_ids)
                for t in tasks[idx]['train']
            ])
            test_inputs = np.array([
                one_hot_channels(t['input'], max_size=num_semantic_ids)
                for t in tasks[idx]['test']
            ])

            try:
                test_outputs = np.array([
                    one_hot_channels(t['output'], max_size=num_semantic_ids)
                    for t in tasks[idx]['test']
                ])
            except KeyError as _e:
                test_outputs = test_inputs

            return dict(
                train_inputs=train_inputs.astype(np.float32),
                train_outputs=train_outputs.astype(np.float32),
                test_inputs=test_inputs.astype(np.float32),
            ), test_outputs.astype(np.float32),

    dl = td.DataLoader(Dataset(), batch_size=bs, shuffle=shuffle)

    return dl


def load_data():
    TRAIN = load_folder('.data/training')
    TRAIN = normalize_obj(TRAIN)
    train_vals = list(TRAIN.values())
    TRAIN_DL = lambda bs, shuffle: get_tasks_dl(train_vals, bs, shuffle)

    VAL = load_folder('.data/evaluation')
    VAL = normalize_obj(VAL)
    val_vals = list(VAL.values())
    VAL_DL = lambda bs, shuffle: get_tasks_dl(val_vals, bs, shuffle)

    return TRAIN_DL, VAL_DL


if __name__ == '__main__':
    print(len(TRAIN), len(VAL))

    print(TRAIN[0]['train'][0]['input'].shape)
