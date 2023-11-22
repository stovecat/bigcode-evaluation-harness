import json
import pickle


def raw_open_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def dump_pkl(data, path):
    with open(path, "wb") as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


# +
def load_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def load_pkl(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data
    
