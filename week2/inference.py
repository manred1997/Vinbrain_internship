import time
import os
import argparse

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer


def load_model(dir_model='./tinybert/checkpoint-100600', dir_tokenizer='./tinybert/tokenizer'):
    print("Loading ..............")
    model = AutoModelForMaskedLM.from_pretrained(dir_model)
    # print("------------Architecture of model tiny bert-------------")
    # print(model)
    # print(f"Number of Parameters:{model.num_parameters()}") #14381394

    tokenizer = BertTokenizer.from_pretrained(dir_tokenizer)
    return model, tokenizer


def inference(string):
    print("Strart inference")
    start = time.time()
    token_string = tokenizer(string, return_tensors="pt")
    # print(token_string)
    output = model(**token_string)[0][0]
    index_mask = np.where(token_string["input_ids"][0].numpy() == 103)[0]
    # print(type(index_mask))
    # print(index_mask)
    results = []
    for i in index_mask:
        output_ids = torch.argmax(output[i])
        # print(output_ids)
        results.append(tokenizer.convert_ids_to_tokens(int(output_ids)))
    print(f"Total time inference : {time.time()-start}")
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_model", default="./tinybert/checkpoint-100600", type=str, help="Model pretrained for MLM")
    parser.add_argument("--dir_tokenizer", default="./tinybert/tokenizer", type=str, help="Tokenizer for MLM")

    args = parser.parse_args()

    model, tokenizer = load_model(args.dir_model, args.dir_tokenizer)

    inputs = "small bilateral pleural [MASK]." # effusions
    results = inference(inputs)
    print(results)

