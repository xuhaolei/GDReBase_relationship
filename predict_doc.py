import pandas as pd
import os

from predict import PredictDisease,PredictBacterium

# step 1. 训练模型
###########################
# step 2. 运行predict.py

df = pd.read_excel('paper_data/paper.xlsx')

disease = PredictDisease()
bacterium = PredictBacterium()

if not os.path.exists('tsv'):
    os.mkdir('tsv')

for i in range(len(df)):
    if os.path.exists('tsv/%05d.txt'%i):
        continue
    d_y_hat,words,d_entities,d_pre_entities = disease.predict(df['abstract'][i])
    b_y_hat,words,b_entities,b_pre_entities = bacterium.predict(df['abstract'][i])
    if i % 100 == 0:
        print(i)

    output = ''
    for j,word in enumerate(words):
        if b_y_hat[j] == 1:
            output += word + '\t' + 'Bacterium' + '\n'
        elif d_y_hat[j] == 1:
            output += word + '\t' + 'Disease' + '\n'
        else:
            output += word + '\t' +'O' +'\n'
    output += '\n' #一个换行表示一个文章结束
    with open('tsv/%05d.txt'%i,'w',encoding='utf-8') as f:
        f.write(output)


    
