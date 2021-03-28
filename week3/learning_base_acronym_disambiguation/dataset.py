import json
import os

from torch.utils.data import Dataset
import torch

from Sample import create_examples
from preprocessing import preprocessing, create_inputs_targets
from create_negative_sample import negative_data


def load_json_file(path: str):
    if not os.path.isfile(path):
        print("Not exist file")
        return None
    with open(path, "r", encoding="UTF-8") as file:
        data = json.load(file)
    return data


class AcrDataset(Dataset):
    def __init__(self, examples, mode="train"):
        self.examples = examples
        self.mode = mode
        if self.mode == "train":
            self.X, self.Y = create_inputs_targets(self.examples, self.mode)
        else:
            self.X, self.Y = create_inputs_targets(self.examples, self.mode)
            
    def __len__(self):
        return self.X[0].shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            input_ids = torch.tensor(self.X[0][idx], dtype=torch.int64)
            input_type_ids = torch.tensor(self.X[1][idx], dtype=torch.int64)
            attention_mask = torch.tensor(self.X[2][idx], dtype=torch.float)
            start_token_idx = torch.tensor(self.Y[0][idx], dtype=torch.int64)
            end_token_idx = torch.tensor(self.Y[1][idx], dtype=torch.int64)
            label = torch.tensor(self.Y[2][idx], dtype=torch.float)
            return input_ids, input_type_ids, attention_mask, start_token_idx, end_token_idx, label
        else:
            input_ids = torch.tensor(self.X[0][idx], dtype=torch.int64)
            input_type_ids = torch.tensor(self.X[1][idx], dtype=torch.int64)
            attention_mask = torch.tensor(self.X[2][idx], dtype=torch.float)
            ids = self.X[3][idx]
            start_token_idx = torch.tensor(self.Y[0][idx], dtype=torch.int64)
            end_token_idx = torch.tensor(self.Y[1][idx], dtype=torch.int64)
            expansion = self.Y[2][idx]
            label = torch.tensor(self.Y[3][idx], dtype=torch.float)
            return input_ids, input_type_ids, attention_mask, ids, start_token_idx, end_token_idx, expansion, label
