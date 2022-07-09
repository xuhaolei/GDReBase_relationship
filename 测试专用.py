from collections import Counter
from nltk.stem import WordNetLemmatizer
import nltk
import os

trigger_word_type = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
trigger_word = []
trigger_word_before = []
trigger_word_middle = []
trigger_word_after = []

def word_fre(file_name, trigger_word_type):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 这个超下标bug找了半天，给的样本tsv有空行，需要判断每一行的长度
        entities = [line.split('\t')[0] for line in lines if len(line.split('\t')) == 2]
        entity_tags = [line.split('\t')[1].replace('\n', '') for line in lines if len(line.split('\t')) == 2]

    window_len = 1
    items = list(zip(entities, entity_tags))
    index = [-1]  # 句点下标，第一句前面没句号，假象一个，下标为-1

    for i, item in enumerate(items):
        if '.' in item:
            index.append(i)
    # print(index)

    # count = 0
    begin = 0
    Disease = False
    Bacterium = False
    while begin + window_len < len(index):
        # 先判断窗口里面是否同时出现实体
        # 断句开始是index[begin]+1,结束是index[begin+window_len]
        indication_pos = []
        for i in range(index[begin] + 1, index[begin + window_len]):
            if 'Disease' in items[i][1]:
                Disease = True
                indication_pos.append(i)
            if 'Bacterium' in items[i][1]:
                Bacterium = True
                indication_pos.append(i)
        if Disease and Bacterium:
            sentence = []
            for i in range(index[begin] + 1, index[begin + window_len]):
                sentence.append(items[i][0])
            tags = nltk.pos_tag(sentence)
            wnl = WordNetLemmatizer()
            for j, tag in enumerate(tags):
                # print(tag[0], tag[1])
                if tag[1] in trigger_word_type:
                    try:
                        # print(wnl.lemmatize(tag[0], tag[1][0].lower()))
                        tag_temp = wnl.lemmatize(tag[0], tag[1][0].lower())
                        trigger_word.append(tag_temp)
                        if j < indication_pos[0]:
                            trigger_word_before.append(tag_temp)
                        elif j > indication_pos[len(indication_pos)-1]:
                            trigger_word_after.append(tag_temp)
                        else:
                            trigger_word_middle.append(tag_temp)
                    except Exception as e:
                        print(e)


        begin += 1
        Disease = False
        Bacterium = False


file_dir = r"tsv"

all_file = os.listdir(file_dir)
for i in range(0, len(all_file)):
    file_name = all_file[i]
    if (i + 1) % 100 == 0:
        print(file_name, i)
    word_fre(file_dir + '\\' + file_name, trigger_word_type)
print("ok!")

trigger_words = [trigger_word,trigger_word_before,trigger_word_middle,trigger_word_after]
trigger_words2 = []
for tw in trigger_words:
    trigger_words2.append(dict(Counter(tw)))
with open('trigger_words.txt','w',encoding='utf-8') as f:
    f.write(str(trigger_words2))
