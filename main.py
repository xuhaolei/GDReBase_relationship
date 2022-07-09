# -*- coding:utf-8 -*-
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from options import args
from dataset import NCBIDataset,Dataset,collate_fn
from models import BiLSTM_CRF
from utils import init_network,evaluate

if __name__ == '__main__':
    device = torch.device('cpu' if args.gpu==-1 else 'cuda:%d'%(args.gpu))
    print('loadind dataset...')
    # step 1. 加载数据集
    if args.dataset == 'ncbi_disease':
        train_set,dev_set,test_set = NCBIDataset(args.train_path,args),\
                                 NCBIDataset(args.dev_path,args),\
                                 NCBIDataset(args.test_path,args)
    else:
        train_set,dev_set,test_set = Dataset(args.train_path,args),\
                                 Dataset(args.dev_path,args),\
                                 Dataset(args.test_path,args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True,collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True,collate_fn=collate_fn)
    
    # step 2. 加载网络模型，并进行初始化，要是有上一轮训练的模型就接着训练
    model = BiLSTM_CRF(args).to(device)
    if os.path.exists(args.save_path): # 接着上一次训练
        model.load_state_dict(torch.load(args.save_path))
    else:
        init_network(model)

    # step 3. 定义优化器(adam)
    # params = list(model.named_parameters())
    # optimizer = torch.optim.Adam([{'params': [p for n, p in params if n != 'embedding.weight']},\
    #                                   {'params':  [p for n, p in params if n == 'embedding.weight'],'lr':1e-5}],\
    #                                    lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # step 4. 定义损失函数(BCE_Loss_function)
    if args.loss_function == 'WBCE':
        def criterions(y_pred, y_true, weight=None, alpha=0.7, gamma=2):
            # sigmoid_p = nn.Sigmoid()(y_pred)
            sigmoid_p = y_pred
            zeros = torch.zeros_like(sigmoid_p)
            pos_p_sub = torch.where(y_true > zeros,y_true - sigmoid_p,zeros)
            neg_p_sub = torch.where(y_true > zeros,zeros,sigmoid_p)
            per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p,1e-8,1.0))-(1-alpha)*(neg_p_sub ** gamma)*torch.log(torch.clamp(1.0-sigmoid_p,1e-8,1.0))
            return per_entry_cross_ent.sum()

    elif args.loss_function == 'BCE':
        criterions = nn.BCELoss()
    print('training...')

    # step 5. 定义flag
    Precision,Recall,F1,Accuracy,Precision_E,Recall_E,F1_E,Accuracy_E = evaluate(model, dev_loader, criterions,device) # 防止之前训练的模型被坏的模型冲掉
    best_f1 = F1 # 当前最好f1
    if best_f1 != best_f1: # nan
        best_f1 = 0
    improve = 0        # 没达到最好f1的轮数
    print('best_f1:%.2f'%best_f1)
    # step 6. 训练模型
    for epoch in range(args.num_epochs):
        if args.testonly: continue
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        now = time.time()
        for sentences,labels,lengths,labels_m,tags,chars in train_loader:
            # step 6.1 将数据放到设备上(cpu or gpu)
            sentences,labels,lengths,labels_m,tags,chars = \
                sentences.to(device),labels.to(device),\
                lengths.to(device),labels_m.to(device),tags.to(device),chars.to(device)

            # step 6.2 正向传播，从模型得到结果output
            model.train()
            model.zero_grad()
            # outputs = model(sentences)
            # [batch_size,seq_len]
            loss = model.neg_log_likelihood_loss(sentences,labels,labels_m,tags,chars) #字典 词性都放在里面了

            # step 6.3 反向传播，计算损失，梯度求导
            # loss = criterions(outputs, labels.float())
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

        # 6.4 每轮后用进行评估
        # train_macro, train_micro, train_loss = evaluate(model, train_loader, criterions,device)
        Precision,Recall,F1,Accuracy,Precision_E,Recall_E,F1_E,Accuracy_E = evaluate(model, dev_loader, criterions,device)
        msg = 'epoch: {0:>6}, Dev P: {1:>6.2%}, Dev R: {2:>6.2%}, Dev F1: {3:>6.2%},Dev A: {4:>6.2%}\n' \
               '              Dev P_E: {5:>6.2%}, Dev R_E: {6:>6.2%}, Dev F1_E: {7:>6.2%},Dev A_E: {8:>6.2%} Time: {9}s'
        print(msg.format(epoch+1, Precision,Recall,F1,Accuracy, Precision_E,Recall_E,F1_E,Accuracy_E, time.time() - now))
        if F1 > best_f1:
            improve = 0
            best_f1 = F1
            # 保存模型
            torch.save(model.state_dict(), args.save_path)
        else:
            improve += 1
            # 早停机制
            if improve > args.patience:
                break

    # step 7. 利用最好模型得到测试集结果
    print('testing...')
    model.load_state_dict(torch.load(args.save_path))
    # torch.save(model.state_dict(), args.save_path) # 保存整个模型
    Precision,Recall,F1,Accuracy,Precision_E,Recall_E,F1_E,Accuracy_E = evaluate(model, test_loader, criterions,device)
    msg = 'Dev P: {0:>6.2%}, Dev R: {1:>6.2%}, Dev F1: {2:>6.2%},Dev A: {3:>6.2%}\n' \
           '              Dev P_E: {4:>6.2%}, Dev R_E: {5:>6.2%}, Dev F1_E: {6:>6.2%},Dev A_E: {7:>6.2%}'
    print(msg.format(Precision,Recall,F1,Accuracy, Precision_E,Recall_E,F1_E,Accuracy_E))
    # msg = 'Test P: {0:>6.2%}, Test R: {1:>6.2%}, Test F1: {2:>6.2%}, Test A: {3:>6.2%}'
    # print(msg.format(Precision,Recall,F1,Accuracy))
