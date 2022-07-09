from collections import Counter
import nltk

file_name = r'C:\Users\徐皓雷\Desktop\paper-ner-master\paper-ner-master\data\BC5CDR-IOB\train.tsv'  # 训练得到的数据
# file_name = r"C:\Users\徐皓雷\Desktop\已经标注的测试集\Bacillus cereus food.tsv"
with open(file_name, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # 这个超下标bug找了半天，给的样本tsv有空行，需要判断每一行的长度
    entities = [line.split('\t')[0] for line in lines if len(line.split('\t')) == 2]
    entity_tags = [line.split('\t')[1].replace('\n', '') for line in lines if len(line.split('\t')) == 2]

window_len = 3
items = list(zip(entities, entity_tags))
index = [-1]  # 句点下标，第一句前面没句号，假象一个，下标为-1

for i, item in enumerate(items):
    if '.' in item:
        index.append(i)
# print(index)


begin = 0
flag_B_Disease = False
flag_I_Disease = False
while begin + window_len < len(index):
    # 先判断窗口里面是否同时出现实体
    # 断句开始是index[begin]+1,结束是index[begin+window_len]
    for i in range(index[begin] + 1, index[begin + window_len]):
        if 'B-Disease' in items[i][1]:
            flag_B_Disease = True
        if 'I-Disease' in items[i][1]:
            flag_I_Disease = True
    if flag_B_Disease and flag_I_Disease:
        sentence = []
        for i in range(index[begin] + 1, index[begin + window_len]):
            sentence.append(items[i][0])
            # 对sentence进行标注
        tags = nltk.pos_tag(sentence)
        tags = [tag[1] for tag in tags]
        tagsCounter = Counter(tags)
        print(dict(tagsCounter))
        if 'VB' in dict(tagsCounter):
            print(dict(tagsCounter)['VB'])
        else:
            print("NO!")
        # print(sentence)
    begin += 1
    flag_B_Disease = False
    flag_I_Disease = False
