import json


def save_dict_to_json(path_file, data_dict):
    with open(path_file, 'w') as fp:
        json.dump(data_dict, fp)


def load_dict_from_json(path_file):
    with open(path_file) as json_file:
        data = json.load(json_file)
    return data


def read_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

