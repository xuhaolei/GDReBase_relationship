# -*- coding:utf-8 -*-
import time
import os
import pickle
import re
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import nltk
from nltk.tokenize import RegexpTokenizer

from options import args
from models import BiLSTM_CRF
from match import DiseaseMatch,BacteriumMatch
from to_word import Atten2Docx

class PredictDisease(object):
    """docstring for Predict"""
    def __init__(self):
        super(PredictDisease, self).__init__()
        # 原来模型的一些参数
        self.args = args
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda:%d'%args.gpu)
        # 词表地址
        self.vocab_path = args.vocab_path
        # 模型地址
        self.model_path = args.save_path
        # 分词
        self.tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
        with open(args.tag_vocab_path,'rb') as f:
            self.tag_vocab = pickle.load(f)
        self.model = BiLSTM_CRF(args).to(self.device)
        self.model.load_state_dict(torch.load(args.save_path,map_location=lambda storage, loc: storage))
        with open(self.vocab_path,'rb') as f:
            self.vocab = pickle.load(f)
        Match = DiseaseMatch
        self.match = Match()
        self.match.build_tree()

    def predict(self,content):
        content = content.strip().lower().replace(',',' , ').replace('.',' . ')
        words = self.tokenizer.tokenize(content)
        
        # 此时分成疾病细菌两部分
        token_ids = [self.vocab[word] if word in self.vocab\
                                 else len(self.vocab)+1 for word in words]
        labels_match,_,entities = self.match.paper_entities(content)
        seq_len = len(token_ids)
        if seq_len == 0: return None,None,None
        labels_one_hot_match = [0] * seq_len # 0 -> other, 1 -> disease 2->begin 3->end
        for label in labels_match:
            for i in range(label[0],label[1]+1):
                labels_one_hot_match[i] = 1
        
        tags = nltk.pos_tag(words)
        tags = [self.tag_vocab.get(t[1],self.tag_vocab.get('<UNK>')) for t in tags]
        
        output,_ = self.model(torch.LongTensor(token_ids).unsqueeze(0).to(self.device),\
                            torch.LongTensor(labels_one_hot_match).unsqueeze(0).to(self.device),\
                            torch.LongTensor(tags).unsqueeze(0).to(self.device))
        y_hat = output.data.cpu().numpy().copy()
        y_hat[y_hat>0] = 1
        y_hat = y_hat.tolist()[0]
        pre_entities = []
        entity = [] 
        for label,word in zip(y_hat,words):
            if label == 0 and len(entity)!=0:
                pre_entities.append(entity)
                entity = []
            if label == 1:
                entity.append(word)

        return y_hat,words,entities,pre_entities


class PredictBacterium(object):
    """docstring for Predict"""
    def __init__(self):
        super(PredictBacterium, self).__init__()
        self.tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
        Match = BacteriumMatch
        self.match = Match()
        self.match.build_tree()

    def predict(self,content):
        content = content.strip().lower().replace(',',' , ').replace('.',' . ')
        words = self.tokenizer.tokenize(content)

        labels_match,_,entities = self.match.paper_entities(content)
        seq_len = len(words)
        if seq_len == 0: return None,None,None
        y_hat = [0] * seq_len # 0 -> other, 1 -> disease 2->begin 3->end
        for label in labels_match:
            for i in range(label[0],label[1]+1):
                y_hat[i] = 1
        return y_hat,words,entities,entities

# label_dic = ['O','X','X']
# p = PredictBacterium()
# with open('paper_data/train_disease.csv','r',encoding='utf-8') as f:
#     lines = csv.reader(f)
#     next(lines)
#     for cnt,line in enumerate(lines):
#         y_hat,words,entities,pre_entities = p.predict(line[2])
#         print(entities)
#         for j,word in enumerate(words):
#             if y_hat[j] == 1:
#                 print(word)
            