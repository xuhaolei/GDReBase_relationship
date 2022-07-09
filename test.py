# -*- coding:utf-8 -*- 
import glob

from nltk.tokenize import RegexpTokenizer
import nltk

tokenizer = RegexpTokenizer(r'[0-9a-zA-Z\'\.,]+')

tagged = nltk.pos_tag(['hello','world'])
print(tagged)

files = glob.glob('files/*.txt')
tag_dic = {}
import time
tag_dic['<PAD>'] = 0

for file in files:
	pre_len = len(tag_dic)
	with open(file,'r',encoding='utf-8') as f:
		paper = f.read()
		tokens = tokenizer.tokenize(paper)
		tagged = nltk.pos_tag(tokens)
		for tag in tagged:
			if tag_dic.get(tag[1]) is None:
				tag_dic[tag[1]] = len(tag_dic)+1
	
	if len(tokens) > 1000 and len(tag_dic) == pre_len:
		break

tag_dic['<UNK>'] = len(tag_dic)+1

import pickle
with open('tag_dic.pkl','wb') as f:
	pickle.dump(tag_dic,f)			

with open('tag_dic.pkl','rb') as f:
	d = pickle.load(f)
print(d)