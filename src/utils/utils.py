import pickle


def read_dataset(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin")
    return data