# -*- coding:utf-8 -*-
import glob
import csv

from itsdangerous import encoding

from nltk.stem import WordNetLemmatizer

def extract_relation(q,sentences,relation_mode):
    dic = {'trigger':[],'Bacterium':[],'Disease':[]}
    for item in q: # item：window:0,i:1,word:2,'trigger':3
        dic[item[3]].append(item)
    # 计算位置平均值
    if len(dic['Bacterium']) == 0 or len(dic['Disease']) == 0:
        return []
    
    relations = []
    disease_list = [] # (name,pos)
    bacterium_list= [] # (name,pos)
    
    pre = -2
    entity = ''
    pos = 0
    cnt = 0
    for d in dic['Disease']:
        # window:0,i:1,word:2,'trigger':3
        if pre + 1 == d[1]:
            entity += d[2].strip() + ' '
        else:
            if cnt != 0:
                disease_list.append((entity.strip(),pos/cnt))
            entity = ''
            cnt = 0
            pos = 0
        
        cnt += 1
        pos += d[1]
        entity += d[2]
        pre = d[1]
    disease_list.append((entity.strip(),pos/cnt))

    
    pre = -2
    entity = ''
    pos = 0
    cnt = 0
    for b in dic['Bacterium']:
        # window:0,i:1,word:2,'trigger':3
        if pre + 1 == b[1]:
            entity += b[2].strip() + ' '
        else:
            if cnt != 0:
                bacterium_list.append((entity.strip(),pos/cnt))
            entity = ''
            cnt = 0
            pos = 0
        
        cnt += 1
        pos += b[1]
        entity += b[2]
        pre = b[1]
    
    bacterium_list.append((entity.strip(),pos/cnt))

    for d in disease_list:
        for b in bacterium_list:
            if d[1] < b[1]:
                begin_entity,end_entity = d[0],b[0]
            else:
                begin_entity,end_entity = b[0],d[0]

            for r in dic['trigger']: # beg, mid ,end
                rmode = relation_mode[r[2]]
                # 三种情况
                if (r[1] <min(d[1],b[1]) and 'beg' in rmode[0]) or \
                    (r[1] >= min(d[1],b[1]) and r[1] <= max(d[1],b[1]) and 'mid' in rmode[0]) or\
                    (r[1] > max(d[1],b[1]) and 'end' in rmode[0]):
                    relations.append([begin_entity,end_entity,r[2],rmode[1],' '.join(sentences)])
    return relations


relation_mode = {}
with open('trigger_word.csv','r',encoding='utf-8') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        # word:0,position:1,relationship:2
        relation_mode[line[0]] = [line[1],line[2]]

files = glob.glob('tsv/*.txt')
window_size = 1
window_current = 0
sentences = []
sentence  = ''
wnl = WordNetLemmatizer()
q = [] # 当成队列用
relations = []
with open('relations.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['entity1', 'entity2','trigger_word' , 'relation_type','sentences'])


for file_cnt,file in enumerate(files):
    with open(file,'r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            line = line.strip().split('\t')
            if ''.join(line) == '': continue
            sentence += line[0] + ' '
            if line[1] != 'O':
                q.append((window_current,i,line[0],line[1])) # 加入实体
            elif wnl.lemmatize(line[0].lower(), 'v') in relation_mode:
                q.append((window_current,i,wnl.lemmatize(line[0].lower(), 'v'),'trigger'))
            if line[0] == '.':
                sentences.append(sentence)
                delj = -1
                for j in range(len(q)):
                    if q[j][0] <= window_current - window_size:
                        delj = j
                    else: break
                if delj > 0: del q[:delj+1] #
                if len(sentences) > window_size: del sentences[0]
                rs = extract_relation(q,sentences,relation_mode)
                if len(rs)>0:
                    with open('relations.csv', 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        for r in rs:
                            csv_writer.writerow(r)
                    relations.extend(rs)
                sentence = ''
                window_current += 1
    if (file_cnt+1) % 100 == 0:
        print(file_cnt + 1,'files have been processed.')


