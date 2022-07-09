# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod

import csv
import glob
import re

from nltk.tokenize import RegexpTokenizer


# paper = """
# Multicopper oxidases (MCOs) waardenburg_syndrome_type_4c represent a diverse family of enzymes that catalyze 
# the oxidation of either an organic or a metal substrate with concomitant 
# reduction of dioxygen to water. These Eubacterium enzymes contain variable numbers of 
# cupredoxin domains, two, three or six per subunit, and rely on four copper ions, 
# a single type I copper and three additional copper ions organized in a 
# trinuclear cluster (TNC), with one type II and two type III copper ions, to 
# catalyze the reaction. Here, two crystal structures and the enzymatic 
# characterization of Marinithermus hydrothermalis MCO, a two-domain enzyme, are 
# reported. This enzyme decolorizes Congo Red dye at 70°C in the presence of high 
# halide concentrations and may therefore be useful in the detoxification of 
# industrial waste that contains dyes. In two distinct crystal structures, MhMCO 
# forms the trimers seen in other two-domain MCOs, but differs from these enzymes 
# in that four trimers interact to create a dodecamer. This dodecamer of MhMCO 
# forms a closed ball-like structure and has implications for the sequestration of 
# bound divalent metal ions as well as substrate accessibility. In each subunit of 
# the dodecameric structures, a Trp residue, Trp351, located between the type I 
# and TNC sites exists in two distinct conformations, consistent with a potential 
# role in facilitating electron transfer in the enzyme.
# """

class Match(object):
    """docstring for Match"""
    def __init__(self):
        super(Match, self).__init__()
        self.tokenizer_paper = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')
        self.tokenizer_entity = RegexpTokenizer(r'[0-9a-zA-Z\']+')
        self.trie_tree = {} # 字典树
        self.end_sign = '__end__'

        # 加载停用词表 这个词表你们也可以看着删删
        with open('common_word.txt',encoding='utf-8') as f:
            self.common_words = set(f.read().split())

    @abstractmethod
    def build_tree(self):
        pass
    
    def paper_entities(self,paper): # 默认paper是小写
        paper = paper.replace('.',' . ').replace(',',' , ')
        paper_tokens = self.tokenizer_paper.tokenize(paper)
        # paper_tokens_origin = paper_tokens.copy()
        paper_tokens = [token for token in paper_tokens]
        # print(paper_tokens)

        labels = []
        start_idx = -1
        end_idx = -1
        for idx,token in enumerate(paper_tokens):
            if idx<=end_idx: continue # 避免重复加入
            tree_node = self.trie_tree
            token_idx = idx
            first_flag = True
            start_idx = idx
            end_idx = -1

            while token_idx <len(paper_tokens) and \
                 (paper_tokens[token_idx] in tree_node or\
                  paper_tokens[token_idx] in self.common_words):
                
                if paper_tokens[token_idx] in tree_node: # 字符
                    tree_node = tree_node[paper_tokens[token_idx]]
                    if first_flag:
                        start_idx = token_idx
                        first_flag = False
                    if self.end_sign in tree_node:
                        end_idx = token_idx

                token_idx += 1 # 

            if start_idx<=end_idx:
                labels.append((start_idx,end_idx)) # 一个实体开始位置和结束位置

        entities = [paper_tokens[start_idx:end_idx+1] for (start_idx,end_idx) in labels]
        return labels,paper_tokens,entities
        

class DiseaseMatch(Match):
    """docstring for Match"""
    def __init__(self):
        super(DiseaseMatch, self).__init__()
        self.files = glob.glob('disease/*.csv') # 疾病列表

    def __process_line(self,line):
        for disease in line:
            if disease == '': continue
            tree_node = self.trie_tree
            in_flag = False
            disease = disease.lower() # 转为小写
            words = self.tokenizer_entity.tokenize(disease)
            if len(''.join(words))<=3:
                continue

            for word in words:
                if word in self.common_words: continue # 停用词不要
                in_flag = True
                if tree_node.get(word) == None:
                    tree_node[word] = {}
                tree_node = tree_node[word]
            if in_flag:
                tree_node[self.end_sign] = {} # 最后一个单词标注为结束符号

    def build_tree(self):
        print('building tree...')
        for file in self.files:
            with open(file,'r',encoding='gbk') as f:
                lines = csv.reader(f)
                next(lines)
                for line in lines:
                    self.__process_line(line) # 处理一行的疾病数据
        print('done.')

class BacteriumMatch(Match):
    """docstring for DiseaseMatch"""
    def __init__(self):
        super(BacteriumMatch, self).__init__()

    def build_tree(self):
        print('building tree...')
        
        with open('bacterium/bacterias.tsv','r',encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                line = line[0].split('\t')
                bacterium = line[3].lower() # 转为小写
                bacterium = re.sub(r' sp\. .*$', '', bacterium)
                words = self.tokenizer_entity.tokenize(bacterium)
                if len(''.join(words))<=3: continue
                tree_node = self.trie_tree
                in_flag = False
                for word in words:
                    if word in self.common_words: continue # 停用词不要
                    in_flag = True
                    if tree_node.get(word) == None:
                        tree_node[word] = {}
                    tree_node = tree_node[word]
                if in_flag:
                    tree_node[self.end_sign] = {} # 最后一个单词标注为结束符号
        print('done.')

if __name__ == '__main__':
    match = BacteriumMatch()
    match.build_tree()
    cnt = 0
    with open('paper_data/train_disease.csv','r',encoding='utf-8') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            # title0,keywords1,abstract2,label3
            labels,tokens,entities = match.paper_entities(line[2])
            if len(labels)>0:
                print(len(labels),' '.join(tokens),entities)
                cnt+=1
            if cnt > 2:
                break
    print(cnt)

