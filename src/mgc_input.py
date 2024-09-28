import pandas as pd

def read_data():
    train = pd.read_csv("../data/train.txt", delimiter='\t', names=["Title", "Industry", "Genre", "Director", "Plot"])
    test = pd.read_csv("../data/test_no_labels.txt", delimiter='\t', names=["Title", "Industry", "Director", "Plot"])
    return train, test