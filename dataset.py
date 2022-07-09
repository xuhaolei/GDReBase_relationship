# -*- coding:utf-8 -*-
import numpy as np
import csv
import re
import pickle
from importlib import import_module

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import nltk
from nltk.tokenize import RegexpTokenizer

from match import DiseaseMatch,BacteriumMatch
from options import args

def word2char(words,char_vocab,pad_len):
    seq = []
    for word in words:
        word_seq = [char_vocab.get(c,char_vocab['<UNK>']) for c in word]
        if len(word_seq) > pad_len:
            word_seq = word_seq[:pad_len]
        else:
            word_seq = word_seq + [char_vocab['<PAD>']] * (pad_len - len(word_seq))
        seq.append(word_seq)
    return seq


class NCBIDataset(Dataset):
    """docstring for PaperDataset"""
    def __init__(self,file,args):
        super(NCBIDataset, self).__init__()

        print('processing ' + file)
        
        with open(args.vocab_path,'rb') as f:
            vocab = pickle.load(f)
        with open(args.tag_vocab_path,'rb') as f:
            tag_vocab = pickle.load(f)
        with open(args.char_vocab_path,'rb') as f:
            char_vocab = pickle.load(f)

        Match = DiseaseMatch
        match = Match()
        match.build_tree()
        tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
        signs = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\''

        self.data_list = []
        labels_dic = {'O':0,'B-SpecificDisease':1, 'I-SpecificDisease':2, 'B-DiseaseClass':3,'I-DiseaseClass':4,\
                     'B-Modifier':5,'I-Modifier':6,'B-CompositeMention':7, 'I-CompositeMention':8}

        with open(file,'r',encoding='utf-8') as f:
            cnt = 0
            num_paper = 0
            title,abstract,labels = '','',[]
            for line in f.readlines():
                if line == '\n':
                    cnt = 0
                    content = title + abstract
                    labels_new = []
                    if content != '':
                        num_paper += 1
                        cnt2 = 0 # 字符指针计数
                        first = True
                        pre = True
                        for i,c in enumerate(content):
                            if c not in signs and c not in ',.' and not first and pre:
                                cnt2 += 1
                                pre = False
                                labels_new.append(-1)
                            elif c in signs:
                                first = False
                                pre = True
                                labels_new.append(cnt2)
                            elif c in ',.':
                                if pre: cnt2 += 1
                                first = False
                                pre = True
                                labels_new.append(cnt2)
                            else:
                                labels_new.append(-1)

                        content = content.strip().lower().replace(',',' , ').replace('.',' . ')
                        # token_x
                        words = tokenizer.tokenize(content)
                        token_ids = [vocab[word] if word in vocab else len(vocab)+1 for word in words]
                        char_ids = word2char(words,char_vocab,args.char_pad_size)

                        seq_len = len(token_ids)
                        labels_one_hot = ['O']*seq_len

                        tags = nltk.pos_tag(words)
                        tags = [tag_vocab.get(t[1],tag_vocab.get('<UNK>')) for t in tags]

                        labels_match,_,_ = match.paper_entities(content)
                        # print(labels_match)
                        labels_one_hot_match = [0] * seq_len # 0 -> other, 1 -> disease 2->begin 3->end
                        for label in labels_match:
                            for i in range(label[0],label[1]+1):
                                labels_one_hot_match[i] = 1

                        labels = [[i for i in range(l[0],l[1])]+[l[2]] for l in labels]
                        for label in labels:
                            tmp = set()
                            entity_name = label[-1]
                            for i in range(len(label)-1):
                                tmp.add(labels_new[label[i]])
                            label = list(tmp)
                            label.sort()
                            if -1 in label: label.remove(-1)
                            
                            labels_one_hot[label[0]] = 'B-'+entity_name
                            for i in range(1,len(label)):
                                labels_one_hot[label[i]] = 'I-'+entity_name
                        
                        labels_one_hot = [labels_dic[c] for c in labels_one_hot]
                        self.data_list.append((token_ids,labels_one_hot,[seq_len],labels_one_hot_match,tags,char_ids))
                        title,abstract,labels = '','',[]
                        if num_paper % 500 == 0:
                            print('%d has been processed.'%(num_paper))
                elif cnt == 1:
                    title = line.split('|')[2]
                elif cnt == 2:
                    abstract = line.split('|')[2]
                else:
                    line = line.split('\t')
                    labels.append([int(line[1]),int(line[2]),line[4]])
                    
                cnt += 1
        print('done.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item][0],self.data_list[item][1],self.data_list[item][2],\
               self.data_list[item][3],self.data_list[item][4],self.data_list[item][5]

class Dataset(Dataset):
    """docstring for PaperDataset"""
    def __init__(self,file,args):
        super(Dataset, self).__init__()

        print('processing ' + file)

        signs = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\',.+/->='
        Match = DiseaseMatch
        match = Match()
        match.build_tree()
        
        with open(args.vocab_path,'rb') as f:
            vocab = pickle.load(f)
        with open(args.tag_vocab_path,'rb') as f:
            tag_vocab = pickle.load(f)
        with open(args.char_vocab_path,'rb') as f:
            char_vocab = pickle.load(f)
        labels_dic = {'O':0,'B-Disease':1,'I-Disease':2,'E-Disease':3,'S-Disease':4}

        tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,\+/\->=]+')
        self.data_list = []
        with open(file,'r',encoding='utf-8') as f:
            lines = csv.reader(f)
            tokens,labels = [],[]
            for cnt,line in enumerate(lines):
                if len(line) == 0:
                    content = ' '.join(tokens)
                    content = content.strip().lower()
                    words = tokenizer.tokenize(content)
                    token_ids = [vocab[word] if word in vocab else len(vocab)+1 for word in words]
                    char_ids = word2char(words,char_vocab,args.char_pad_size)
                    seq_len = len(token_ids)
                    labels_one_hot = [0] * seq_len
                    assert seq_len == len(labels)

                    tags = nltk.pos_tag(words)
                    tags = [tag_vocab.get(t[1],tag_vocab.get('<UNK>')) for t in tags]

                    for i,label in enumerate(labels):
                        if labels_dic.get(label) is None:
                            labels_dic[label] = len(labels_dic)
                        labels_one_hot[i] = labels_dic[label]

                    labels_match,_,_ = match.paper_entities(content)
                    labels_one_hot_match = [0] * seq_len # 0 -> other, 1 -> disease 2->begin 3->end
                    for label in labels_match:
                        for i in range(label[0],label[1]+1):
                            labels_one_hot_match[i] = 1

                    self.data_list.append((token_ids,labels_one_hot,[seq_len],labels_one_hot_match,tags,char_ids))
                    tokens,labels = [],[]
                
                elif len(line[0].split('\t'))==2 and line[0]!='\n':
                    line = line[0].split('\t')
                    if len(line[0]) == 1 and line[0] not in signs:
                        continue
                    
                    tokens.append(line[0])
                    labels.append(line[1])

                if (cnt+1) % 10000 == 0:
                    print('%d lines has been processed.'%(cnt+1))
        print('done.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item][0],self.data_list[item][1],self.data_list[item][2],\
               self.data_list[item][3],self.data_list[item][4],self.data_list[item][5]


class MatchDataset(Dataset):
    """docstring for PaperDataset"""
    def __init__(self,file,args):
        super(Dataset, self).__init__()

        print('processing ' + file)

        signs = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\',.+/->='
        Match = DiseaseMatch
        match = Match()
        match.build_tree()
        
        with open(args.vocab_path,'rb') as f:
            vocab = pickle.load(f)
        with open(args.tag_vocab_path,'rb') as f:
            tag_vocab = pickle.load(f)

        labels_dic = {'O':0,'B-Disease':1,'I-Disease':2}
        

        tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
        self.data = np.zeros((3)) # TP_FN,TP_TF,TP
        with open(file,'r',encoding='utf-8') as f:
            lines = csv.reader(f)
            tokens,labels = [],[]
            for cnt,line in enumerate(lines):
                if len(line) == 0:
                    content = ' '.join(tokens)
                    content = content.strip().lower()
                    seq_len = len(tokens)
                    labels_one_hot = [0] * seq_len
                    for i,label in enumerate(labels):
                        if labels_dic.get(label) is None:
                            labels_dic[label] = len(labels_dic)
                        labels_one_hot[i] = labels_dic[label]

                    labels_match,_,_ = match.paper_entities(content)
                    labels_one_hot_match = [0] * seq_len # 0 -> other, 1 -> disease 2->begin 3->end
                    for label in labels_match:
                        for i in range(label[0],label[1]+1):
                            labels_one_hot_match[i] = 1

                    labels_one_hot = np.array(labels_one_hot)
                    labels_one_hot_match = np.array(labels_one_hot_match)

                    labels_one_hot_match[labels_one_hot_match>0] = 1
                    labels_one_hot[labels_one_hot>0] = 1
                    
                    self.data[0] += np.sum(labels_one_hot)
                    self.data[1] += np.sum(labels_one_hot_match)
                    self.data[2] += np.sum(labels_one_hot*labels_one_hot_match)

                    tokens,labels = [],[]
                
                elif len(line[0].split('\t'))==2 and line[0]!='\n':
                    line = line[0].split('\t')
                    if len(line[0]) == 1 and line[0] not in signs:
                        continue
                    
                    tokens.append(line[0])
                    labels.append(line[1])

                if (cnt+1) % 10000 == 0:
                    print('%d lines has been processed.'%(cnt+1))
        print('done.')
        P = self.data[2]/(self.data[1]+1e-5)
        R = self.data[2]/(self.data[0]+1e-5)
        F1= 2*P*R/(P+R+1e-5)
        print('P:%f,R:%f,F1:%f'%(P,R,F1))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item][0],self.data_list[item][1],self.data_list[item][2],\
               self.data_list[item][3],self.data_list[item][4]

def collate_fn(batch):
    pad_len = 0
    for item in batch:
        pad_len = max(item[2][0],pad_len)
    token_ids,labels,lengths,labels_match,tags,chars = [],[],[],[],[],[]
    for item in batch:
        
        # 按照batch pad
        token_id = item[0] + [0]*(pad_len - item[2][0])
        label = item[1] + [0]*(pad_len - item[2][0])
        label_match = item[3] + [0]*(pad_len - item[2][0])
        tag = item[4] + [0] * (pad_len-item[2][0])
        char = item[5] + [[0] * args.char_pad_size] * (pad_len-item[2][0])

        token_ids.append(torch.tensor(token_id).unsqueeze(0))
        labels.append(torch.tensor(label).unsqueeze(0))
        lengths.append(torch.tensor(item[2]).unsqueeze(0))
        labels_match.append(torch.tensor(label_match).unsqueeze(0))
        tags.append(torch.tensor(tag).unsqueeze(0))
        chars.append(torch.tensor(char).unsqueeze(0))


    token_ids = torch.cat(token_ids,dim = 0)
    labels = torch.cat(labels,dim = 0)
    lengths = torch.cat(lengths,dim = 0)
    labels_match = torch.cat(labels_match,dim = 0)
    tags = torch.cat(tags,dim = 0)
    chars = torch.cat(chars,dim = 0)

    return token_ids,labels,lengths,labels_match,tags,chars


if __name__ == '__main__':
    pass

    # train_set =  NCBIDataset(args.train_path,args)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size,
    #                          num_workers=1, shuffle=True,collate_fn = collate_fn)
    # for data in train_loader:
    #     print(data)
    #     break

    # train_set = MatchDataset(args.train_path,args) # 检测纯匹配准确率

    # char_vocab = '0123456789abcdefghijklmnopqrstuvwxyz.,'
    # vocab = {}
    # vocab['<PAD>'] = 0
    # for idx,char in enumerate(char_vocab):
    #     vocab[char] = idx+1
    # vocab_size = len(vocab)
    # vocab['<UNK>'] = vocab_size
    # with open('char_vocab.pkl','wb') as f:
    #     pickle.dump(vocab,f)



