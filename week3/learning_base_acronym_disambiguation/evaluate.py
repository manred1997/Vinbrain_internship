import json
import random
import sys

from tqdm import tqdm
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Sample import Sample, create_examples
from preprocessing import create_inputs_targets

from metrics import accracy_score, precision_recall_f1

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import BertWordPieceTokenizer

from model import AcrBertModel

from dataset import AcrDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertWordPieceTokenizer("slow_token/vocab.txt", lowercase=True)

with open("dev_data/dev_data.json", "r") as f:
    dev = json.load(f)
with open("dev_data/dev_neg_data.json", "r") as f:
    dev_neg = json.load(f)

dev.extend(dev_neg)

examples_dev = create_examples(dev, "Create dev examples", tokenizer)

# X, Y = create_inputs_targets(examples_dev, mode="dev")

# dev_data = TensorDataset(torch.tensor(X[0], dtype=torch.int64),
#                             torch.tensor(X[1], dtype=torch.int64),
#                             torch.tensor(X[2], dtype=torch.float),
#                             torch.tensor(Y[0], dtype=torch.int64),
#                             torch.tensor(Y[1], dtype=torch.int64),
#                             torch.tensor(Y[2], dtype=torch.float))

# dev_sampler = SequentialSampler(dev_data)
dev_data = AcrDataset(examples, mode="dev")
dev_data_loader = DataLoader(dev_data, batch_size=16, shuffle="True")

model = AcrBertModel.from_pretrained(pretrained_model_name_or_path="./weights_4.pth",
                                     config="./bert_base/config.json").to(device=device)

for step, batch in enumerate(dev_data_loader):
    batch = tuple(t.to(device=device) for t in batch)
    input_word_ids, input_type_ids, input_mask, ids, start_token_idx, end_token_idx, expansion, label = batch
    output, _ = model(input_ids=input_word_ids,
                        token_type_ids=input_type_ids,
                        attention_mask=input_mask,
                        start_token_idx=start_token_idx,
                        end_token_idx=end_token_idx)
    
        
    








