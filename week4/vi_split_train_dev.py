import json
import argparse
from sklearn.model_selection import train_test_split
import random

def split_train_dev(data: list, test_size=0.1) -> list:
    """
    Function split data
    """
    train, dev = train_test_split(data, test_size=test_size)
    return train, dev

if __name__ == "__main__":
    with open("./data_cxr/pos_data.json", "r", encoding="UTF-8") as f:
        data_pos = json.load(f)
    print(f"Length of pos data {len(data_pos)}")
    with open("./data_cxr/neg_data.json", "r", encoding="UTF-8") as f:
        data_neg = json.load(f)
    print(f"Length of neg data {len(data_neg)}")
    
    data_pos.extend(data_neg)
    random.shuffle(data_pos)

    train, test = split_train_dev(data_pos)

    with open("./data_cxr/train_data.json", "w", encoding="UTF-8") as f:
        json.dump(train, f)
    with open("./data_cxr/test_data.json", "w", encoding="UTF-8") as f:
        json.dump(test, f)
    

