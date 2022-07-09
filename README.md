# paper-ner

## requirements
pytorch
pandas
nltk
gensim

## data source
`data/*.xlsx` : https://gitee.com/loxs/paper-classification

## 运行

step 1. 数据预处理，命令行下：`python utils.py`

step 2. 训练模型，命令行下：`python main.py`
 
可以在options中更换数据集：可供更换的数据集为
data目录下所有文件夹名字，以及ncbi_disease


step 3. 预测，命令行下:`python predict.py`