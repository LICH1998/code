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

logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class Adapter(nn.Module):
    """
    构造adapter layers(bottleneck)结构 - see arXiv:1902.00751.
    高维度映射到低维度->非线性层->低维度再映射到高维度（初始输入到输出加残差连接）
    """

    def __init__(self, input_size, hidden_size=64, init_scale=1e-3):
        super().__init__()
        self.adapter_down = nn.Linear(input_size, hidden_size)
        nn.init.trunc_normal_(self.adapter_down.weight, std=init_scale)
        nn.init.zeros_(self.adapter_down.bias)
        self.adapter_up = nn.Linear(hidden_size, input_size)
        nn.init.trunc_normal_(self.adapter_up.weight, std=init_scale)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x_c = self.adapter_down(x)
        x_c = F.gelu(x_c)
        x_c = self.adapter_up(x_c)
        return x_c + x


class BertClassifier(nn.Module):
    """
    建立adapter-bert分类器
    """

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config['bert_path'], output_hidden_states=True)
        self.add_adapter()  # 加入adapter_layer
        self.dropout = nn.Dropout(self.config['dropout'])
        self.linear768x768 = nn.Linear(768, 768)
        self.linear768x2 = nn.Linear(768, 2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, X):
        hidden_states = self.bert(input_ids=X['input_ids'].squeeze(1), attention_mask=X['attention_mask'], return_dict=False)[2]
        dropout_output1 = self.dropout(hidden_states[-1][:, 0])
        linear_output1 = self.tanh(self.linear768x768(dropout_output1))
        dropout_output2 = self.dropout(linear_output1)
        linear_output2 = self.softmax(self.linear768x2(dropout_output2))

        return dropout_output1, linear_output1, linear_output2

    def add_adapter(self):
        for i in range(12):
            # Layer:12, Hidden:768
            self.bert.encoder.layer[i].attention.output.LayerNorm = nn.Sequential(Adapter(768), self.bert.encoder.layer[i].attention.output.LayerNorm)
            self.bert.encoder.layer[i].output.LayerNorm = nn.Sequential(Adapter(768), self.bert.encoder.layer[i].output.LayerNorm)
        for name, param in self.bert.named_parameters():
            # freeze除了LayerNorm（包括adapter layers）以外的参数
            if not ("LayerNorm" in name):
                print("freezing", name)
                param.requires_grad = False


class FinDataset(torch.utils.data.Dataset):
    """
    构造pytorch所需的Dataset，并对文本进行tokenize
    """

    def __init__(self, df, max_seq_len):
        self.labels = df['MKTSENTIMENTS'].values.tolist()
        print('Start Tokenizing Text ......')
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=max_seq_len,
                                truncation=True,
                                return_tensors="pt") for text in tqdm(df['CONTENT'], position=0, leave=True)]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class GetDataset():
    """
    1.读取数据
    2.切分测试集与训练集
    3.转换为FinDataset对象
    4.转换为DataLoader对象
    """

    def __init__(self, config):
        self.config = config
        df = self.read_df()
        df_train, df_val = self.train_valid_split(df)
        train_dataset = FinDataset(df_train, self.config['max_seq_len'])
        val_dataset = FinDataset(df_val, self.config['max_seq_len'])
        self.train_dataloader, self.val_dataloader = self.get_dataloader(train_dataset, val_dataset)

    def read_df(self):
        """
        读取数据集
        """
        df = pd.read_csv(self.config['data_path']).dropna()
        df.drop(["INDEX", "PUBLISHDATE", "SOURCE", "WINDCODES", "OPDATE"], axis=1, inplace=True)
        return df

    def train_valid_split(self, df):
        """
        打乱数据集，并以split_ratio分割训练集与验证集
        """
        df_train, df_val = np.split(df.sample(frac=1, random_state=self.config['random_state']),
                                    [int(self.config['split_ratio'] * len(df))])
        return df_train, df_val

    def get_dataloader(self, train_dataset, val_dataset):
        """
        生成dataloader对象
        """
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=1)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=1)
        return train_dataloader, val_dataloader


def warm_up_lr(epoch, max_learn_rate=1.5e-5, end_learn_rate=1e-7, warmup_epoch_count=3, total_epoch_count=10):
    if epoch < warmup_epoch_count:
        lr = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
    else:
        lr = max_learn_rate * math.exp(math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (total_epoch_count - warmup_epoch_count + 1))
    return float(lr)


def train(model, train_dataloader, val_dataloader, config):
    device = config['device']
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    for epoch in range(config['epochs']):
        len_train = 0
        acc_num = 0
        total_loss_train = 0
        optimizer = Adam(model.parameters(), lr=warm_up_lr(epoch, config['max_lr'], config['end_lr'], config['warm_up'], config['epochs']))

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}', position=0, leave=True) as _tqdm:
            for train_input, train_label in train_dataloader:
                len_train += len(train_label)
                train_label = train_label.to(device)
                train_input['input_ids'] = train_input['input_ids'].to(device)
                train_input['token_type_ids'] = train_input['token_type_ids'].to(device)
                train_input['attention_mask'] = train_input['attention_mask'].to(device)

                output = model(train_input)[2]

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc_num += (output.argmax(dim=1) == train_label).sum().item()
                _tqdm.set_postfix(train_acc='{:.4f}'.format(acc_num / len_train))

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                _tqdm.update(1)
        # ------ 验证模型 -----------
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            len_val = 0
            for val_input, val_label in val_dataloader:
                len_val += len(val_label)
                val_label = val_label.to(device)
                val_input['input_ids'] = val_input['input_ids'].to(device)
                val_input['token_type_ids'] = val_input['token_type_ids'].to(device)
                val_input['attention_mask'] = val_input['attention_mask'].to(device)

                output = model(val_input)[2]

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'''
              | Train Loss: {total_loss_train / len_train: .3f}
              | Train Accuracy: {acc_num / len_train: .3f}
              | Val Loss: {total_loss_val / len_val: .3f} 
              | Val Accuracy: {total_acc_val / len_val: .3f}\n''')
        torch.save(model.state_dict(), f'../model/FinBERT_finetuning/adapter_bert_{epoch}.torch')


if __name__ == '__main__':

    config = dict()
    config['batch_size'] = 16
    config['max_seq_len'] = 500
    config['split_ratio'] = 0.7
    config['random_state'] = 42
    config['epochs'] = 10
    config['warm_up'] = 3
    config['max_lr'] = 1.5e-5
    config['end_lr'] = 1e-7
    config['dropout'] = 0.5
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['data_path'] = "../temp_data/WIND_FinancialNews_cleaned/processed_data.csv"
    config['bert_path'] = "../model/FinBERT_L-12_H-768_A-12_pytorch"

    tokenizer = BertTokenizer.from_pretrained(config['bert_path'], use_fast=True)

    get_dataset = GetDataset(config)
    train_dataloader, val_dataloader = get_dataset.train_dataloader, get_dataset.val_dataloader
    bert_classifier = BertClassifier(config)
    train(bert_classifier, train_dataloader, val_dataloader, config)