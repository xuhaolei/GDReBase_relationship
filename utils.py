# -*- coding:utf-8 -*-
import pandas as pd
import pickle
import numpy as np
import tqdm
import re
import csv
import sys
import glob
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
from nltk.tokenize import RegexpTokenizer

import gensim.models
import gensim.models.word2vec as w2v
import gensim.models.fasttext as fasttext

from options import args

MAX_SIZE = 2000

def build_vocab(file_path,outfile_path, tokenizer, max_size=200000, min_freq=3):
    vocab_dic = {}
    files = glob.glob(file_path)
    for cnt,file in enumerate(files):
        with open(file,'r',encoding='utf-8') as f:
            content = f.read()
            content = content.replace('.',' . ').replace(',',' , ')
            if not content: continue
            for word in tokenizer.tokenize(content):
                word = word.lower()
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
            if (cnt+1) % 1000 == 0:
                print('%d done.'%(cnt+1))

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx+1 for idx, word_count in enumerate(vocab_list)}
    # vocab_dic.update({UNK: len(vocab_dic), PAD: 0})
    with open(outfile_path,'wb') as f:
        pickle.dump(vocab_dic,f)
    return vocab_dic

def gensim_to_embeddings(wv_file, vocab_file,outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    with open(vocab_file,'rb') as f:
        vocab = pickle.load(f)

    # ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    ind2w = {vocab[key]:key for key in vocab}
    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)


def word_embeddings(out_file,notes_file,tokenizer, embedding_size, min_count, n_iter):

    sentences = ProcessedIter(notes_file,tokenizer)

    model = w2v.Word2Vec(vector_size=embedding_size, min_count=min_count, workers=4, epochs=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index_to_key[0])) ))
    words = ["**PAD**"]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index_to_key[0])))
    for idx, word in ind2w.items():
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w',encoding='utf-8') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

class ProcessedIter(object):

    def __init__(self, files,tokenizer):
        self.files = files # 'files/*.txt'
        self.tokenizer = tokenizer
    def __iter__(self):
        files = glob.glob(self.files)
        for cnt,file in enumerate(files):
            with open(file,'r',encoding='utf-8') as f:
                content = f.read().strip()
                if content == '': continue
                content = content.replace('.',' . ').replace(',',' , ')
                content = self.tokenizer.tokenize(content)
                content = [c.lower() for c in content]
                yield content

def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        #UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_accuracy(true_labels, pred_labels):
    flat_true = true_labels.ravel()    # 将多维数组转换为1维数组
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / union_size(flat_true, flat_pred, 0)


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def union_size(x, y, axis):       # 或逻辑后加和
    return np.logical_or(x, y).sum(axis=axis).astype(float)


def intersect_size(x, y, axis):   # 与逻辑后加和
    return np.logical_and(x, y).sum(axis=axis).astype(float)

def evaluate(model, data_iter, criterions, device):
    model.eval()
    Precision,Recall,F1,Accuracy = 0,0,0,0
    Precision_E,Recall_E,F1_E,Accuracy_E = 0,0,0,0
    with torch.no_grad():

        for sentences,labels,lengths,labels_m,tags,chars in data_iter:
            # step 5.1 将数据放到设备上(cpu or gpu)
            sentences,labels,lengths,labels_m,tags,chars = \
                sentences.to(device),labels.to(device),\
                lengths.to(device),labels_m.to(device),tags.to(device),chars.to(device)

            outputs,_ = model(sentences,labels_m,tags,chars) # labels_m 是字典匹配结果

            y = labels.data.cpu().numpy().copy() # [batch_size, seq_len]
            y_hat = outputs.data.cpu().numpy().copy() # [batch_size, seq_len]
            lengths = lengths.data.cpu().numpy().copy() # [batch_size, 1]
            lengths = lengths.squeeze() # [batch_size]

            y[y>0] = 1
            y_hat[y_hat>0] = 1

            mask = np.zeros((y.shape[0],y.shape[1]))
            for i in range(lengths.shape[0]):
                mask[i,0:lengths[i]] = 1
            
            # for i in range(y.shape[1]):
            # 实体级别 B（词首1）I（词中2）E（词尾3）S（单独成词4）
            flag = True
            y_labels = np.zeros((y.shape[0],y.shape[1]*y.shape[1]))
            start_idx = -1
            end_idx = -1
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if flag and y[i,j] == 1:
                        flag = False
                        start_idx = j
                    if not flag and y[i,j]==0:
                        flag = True
                        end_idx = j-1
                        y_labels[i,start_idx*y.shape[1]+end_idx] = 1 # 映射为一个数字
                    # if y[i,j] == 4:
                    #     y_labels[i,j*y.shape[1]+j] = 1 # 映射为一个数字


            flag = True
            y_hat_labels = np.zeros((y_hat.shape[0],y_hat.shape[1]*y_hat.shape[1]))
            start_idx = -1
            end_idx = -1
            for i in range(y_hat.shape[0]):
                for j in range(y_hat.shape[1]):
                    if flag and y_hat[i,j] == 1:
                        flag = False
                        start_idx = j
                    if not flag and y_hat[i,j]==0:
                        flag = True
                        end_idx = j-1
                        y_hat_labels[i,start_idx*y_hat.shape[1]+end_idx] = 1  # 映射为一个数字
                    # if y[i,j] == 4:
                    #     y_labels[i,j*y.shape[1]+j] = 1 # 映射为一个数字

            TP_TN = np.sum((y==y_hat)*mask)
            TP = np.sum((y == y_hat)*(y>0)*(y_hat>0))
            TP_FN = np.sum(y>0)
            TP_FP = np.sum(y_hat>0)
            ALL = np.sum(lengths)

            eps = 1e-6
            P = TP/(TP_FP+eps)
            R = TP/(TP_FN+eps)
            Precision += P
            Recall += R
            F1 += 2*P*R/(P+R+eps)
            Accuracy += TP_TN / (ALL+eps)


            TP_TN_E = np.sum((y_labels == y_hat_labels))
            TP_E = np.sum((y_labels == y_hat_labels)*(y_labels>0)*(y_hat_labels > 0))
            TP_FN_E = np.sum(y_labels>0)
            TP_FP_E = np.sum(y_hat_labels>0)
            ALL_E = np.sum(lengths)

            eps = 1e-6
            P_E = TP_E/(TP_FP_E+eps)
            R_E = TP_E/(TP_FN_E+eps)
            Precision_E += P_E
            Recall_E += R_E
            F1_E += 2*P_E*R_E/(P_E+R_E+eps)
            Accuracy_E += TP_TN_E / (ALL_E+eps)

    return Precision/len(data_iter),Recall/len(data_iter),F1/len(data_iter),Accuracy/len(data_iter),\
           Precision_E/len(data_iter),Recall_E/len(data_iter),F1_E/len(data_iter),Accuracy_E/len(data_iter)

if __name__ == "__main__":
    # 从原始paper.xlsx中获取数据，放到disease_full.xlsx中

    # get_data('%s_full.xlsx'%dataset,dataset)
    # # 划分训练集，开发集，测试集 -> train_disease.csv,dev_disease.csv,test_disease.csv
    # split_dataset('%s_full.xlsx'%dataset)

    # 构件词表 -> vocab.pkl
    tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
    build_vocab('files/*.txt','vocab.pkl', tokenizer, 5000, 3)

    # # 得到预训练词向量，最终得到processed.embed
    # w2v_file = word_embeddings('word_embed/processed.w2v','files/*.txt',tokenizer, 100, 0, 10)
    gensim_to_embeddings('word_embed/processed.w2v', 'vocab.pkl')