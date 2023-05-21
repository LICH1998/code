# -*- coding: utf-8 -*-
"""
@author: HTSC
"""

import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, logging
from AdapterBert import Adapter, BertClassifier

logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class FinDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_seq_len):
        print('Start Tokenizing Text ......')
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=max_seq_len,
                                truncation=True,
                                return_tensors="pt") for text in tqdm(df['CONTENT'], position=0, leave=True)]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts

def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x['input_ids'] = x['input_ids'].to(device)
        x['token_type_ids'] = x['token_type_ids'].to(device)
        x['attention_mask'] = x['attention_mask'].to(device)
        with torch.no_grad():
            pred = model(x)[0]
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

if __name__ == '__main__':

    config = dict()
    config['batch_size'] = 32
    config['max_seq_len'] = 500
    config['dropout'] = 0.5
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['data_path'] = "../temp_data/data_report_adjust_split_word_allinfo_origin.csv"
    config['bert_path'] = "../model/FinBERT_L-12_H-768_A-12_pytorch"
    config['adapter_bert_path'] = "../model/FinBERT_finetuning/adapter_bert_9.torch"
    config['result_path'] = "../temp_data/data_report_adjust_split_word_allinfo.csv"

    tokenizer = BertTokenizer.from_pretrained(config['bert_path'], use_fast=True)

    df = pd.read_csv(config['data_path'], index_col=0, dtype={'STOCK_CODE': str})
    dataset = FinDataset(df, config['max_seq_len'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], num_workers=1)
    bert_classifier = BertClassifier(config)
    bert_classifier.load_state_dict(torch.load(config['adapter_bert_path']))
    bert_classifier.to(config['device'])
    content_processed = predict(dataloader, bert_classifier, config['device'])
    df['CONTENT_processed'] = content_processed.tolist()
    df.to_csv(config['result_path'])