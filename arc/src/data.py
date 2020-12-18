import os
import json
import numpy as np


def matrix_to_np(obj):
    obj = obj.copy()

    iterator = obj.items() if type(obj) == dict else enumerate(obj)

    for k, v in iterator:
        is_matrix = type(v) == list and type(v[0]) == list
        if is_matrix:
            obj[k] = np.array(v)
        else:
            obj[k] = matrix_to_np(v)

    return obj


def load_folder(path):
    result = []

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            json_content = json.loads(text)
            json_content = matrix_to_np(json_content)
            result.append(json_content)

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


def normalize_matrix_size(matrix):
    matrix = np.pad(
        matrix,
        ((1, 1), (1, 1)),
        'constant',
        constant_values=((10, 10), (10, 10)),
    )
    max_width = 32
    max_height = 32

    h, w = matrix.shape

    padding_x = (max_width - w) // 2
    padding_y = (max_height - h) // 2

    output = np.zeros((max_width, max_height))
    output[
        padding_y: padding_y + h,
        padding_x: padding_x + w
    ] = matrix

    return output


def normalize_obj(obj):
    return map_np(normalize_matrix_size, obj)


TRAIN = load_folder('.data/training')
TRAIN = normalize_obj(TRAIN)

VAL = load_folder('.data/evaluation')
VAL = normalize_obj(VAL)

if __name__ == '__main__':
    print(len(TRAIN), len(VAL))

    print(TRAIN[0]['train'][0]['input'].shape)
