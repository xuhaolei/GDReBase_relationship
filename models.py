# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_embeddings


class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        assert config.embed_file is not None # 断言
        # W = torch.Tensor(load_embeddings(config.embed_file))
        # self.embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embedding = nn.Embedding(5002, config.embed_size, padding_idx=0)
        self.tag_embedding = nn.Embedding(config.tag_vocab_size, config.tag_embed_size, padding_idx=0)
        self.char_embedding = nn.Embedding(config.char_vocab_size,config.char_embed_size,padding_idx=0)
        # self.embedding.weight.data = W.clone() # 
        self.dropout = nn.Dropout(config.dropout)
        self.bi_lstm = nn.LSTM(config.embed_size + config.tag_embed_size + config.char_conv_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(config.char_embed_size,config.char_conv_size,3,padding = 1)
        self.fc = nn.Linear(2*config.hidden_size, config.target_size)
        # self.fc1 = nn.Linear(4*config.hidden_size, config.ffn_size)
        # self.fc2 = nn.Linear(config.ffn_size, config.target_size)
        # 3 -> B I O 三种状态
        self.A = nn.Parameter(torch.randn(config.target_size,config.target_size)) # 转移矩阵CRF层用
        self.wd = nn.Parameter(torch.randn(1,1,2*config.hidden_size))
        # self.wd = nn.Parameter(torch.randn(1))

    def _log_sum_score(self,alpha,dim):
        # alpha  : [batch_size, 3, 3] or [batch_size, 3]
        # return : [batch_size, 3]    or [batch_size   ]
        maxv,_ = alpha.max(dim = dim,keepdim = True)
        return torch.log(torch.exp(alpha - maxv).sum(dim = dim)) + maxv.squeeze()

    def _sum_score(self,X):
        #    X   : [batch_size, seq_len, target_size] ∈ R
        # return : [batch_size] ∈ R
        batch_size = X.size(0)
        seq_len = X.size(1)
        target_size = X.size(2) # 一般为3 BIO

        alpha = X[:,0,:] # 最开始的发射矩阵
        for i in range(seq_len-1):
            alpha = alpha.unsqueeze(1).expand(batch_size,target_size,target_size)

            alpha = alpha.transpose(1,2) + self.A.unsqueeze(0) + \
                    X[:,i+1,:].unsqueeze(1).expand(batch_size,target_size,target_size)

            alpha = self._log_sum_score(alpha,1) # [batch_size, 3]

        alpha = self._log_sum_score(alpha,1) # [batch_size]
        return alpha


    def _score(self,X,Y): # 计算Y对应的分数
        #    X   : [batch_size, seq_len, 3] ∈ R
        #    Y   : [batch_size, seq_len   ] ∈ {0,1,2}
        # return : [batch_size       ] ∈ R
        batch_size = X.size(0)
        seq_len = X.size(1)
        target_size = X.size(2)

        # score_p : [batch_size,seq_len]
        score_p = X.view(-1,target_size)[np.arange(batch_size*seq_len),Y.view(-1)].view(batch_size,-1)
        score_p = torch.sum(score_p,dim = 1)

        yfrom = Y[:,:-1].contiguous().view(-1)
        yto = Y[:,1:].contiguous().view(-1)
        score_a = self.A[yfrom,yto].view(batch_size,-1)
        score_a = torch.sum(score_a,dim = 1)
        return score_p + score_a

    def _lstm_features(self,X,labels_m,tags,chars):
        # labels_m : [batch_size,seq_len] ∈ {0,1}
        batch_size = X.size(0)
        seq_len = X.size(1)
        word_len = chars.size(2)

        embed = self.embedding(X) # [batch_size,seq_len,embed_size]
        tag_embed = self.tag_embedding(tags)

        char_embed = self.char_embedding(chars)  # [batch,seq_len,word_len,embed_size]
        char_embed = char_embed.permute(0,3,1,2) # [batch,embed_size,seq_len,word_len]
        char_embed = char_embed.view(batch_size,-1,seq_len*word_len)
        char_conv = self.conv(char_embed) # [batch,30,seq_len*word_len]
        char_conv,_ = char_conv.view(batch_size,-1,seq_len,word_len).max(dim = 3)
        char_conv = char_conv.permute(0,2,1)

        embed = torch.cat((embed,tag_embed,char_conv),dim = 2)
        embed = self.dropout(embed) # [batch_size, seq_len, embed_size]
        H,(_,_) = self.bi_lstm(embed) # [batch_size, seq_len, 2*hidden_size]
        H = H + labels_m.unsqueeze(2) * self.wd
        P = self.fc(H)
        return P
        
    # paper : https://doi.org/10.1016/j.compbiomed.2019.04.002
    # def _lstm_features(self,X,labels_m,tags):
    #     # labels_m : [batch_size,seq_len] ∈ {0,1}
    #     seq_len = X.size(1)

    #     embed = self.embedding(X) # [batch_size,seq_len,embed_size]
    #     tag_embed = self.tag_embedding(tags)
    #     embed = torch.cat((embed,tag_embed),dim = 2)
    #     embed = self.dropout(embed) # [batch_size, seq_len, embed_size]

    #     H,(_,_) = self.bi_lstm(embed) # [batch_size, seq_len, 2*hidden_size]

    #     # 广义发射概率是对数形式 ∈ [-无穷,+无穷]
    #     P = self.fc(H) # [batch_size, seq_len, 3] 'B I O'
    #     L2 = torch.norm(H,dim = 2) # [batch_size, seq_len]
    #     P = []
    #     for i in range(seq_len):
    #         S_i = H[:,i,:].unsqueeze(1) @ H.transpose(1,2) / (L2[:,i].unsqueeze(1) * L2).unsqueeze(1)# [batch_size,1,seq_len]

    #         S_i = (self.wd * labels_m[:,i]).unsqueeze(1) + (1-self.wd) * S_i.squeeze(1) # [batch_size,seq_len]
            
    #         S_i = (torch.softmax(S_i,dim = 1).unsqueeze(1) @ H).squeeze() # [batch_size,2 * hidden_size]

    #         G_i = torch.cat((S_i,H[:,i,:]),dim = 1)

    #         Z_i = torch.tanh(self.fc1(G_i))

    #         P_i = torch.tanh(self.fc2(Z_i))

    #         P.append(P_i.unsqueeze(1))

    #     P = torch.cat(P,dim = 1) # [batch_size,seq_len,target_size]
    #     return P


    def neg_log_likelihood_loss(self,X,Y,labels_m,tags,chars):
        P = self._lstm_features(X,labels_m,tags,chars)

        sum_score = self._sum_score(P)
        gold_score = self._score(P,Y)
        return torch.mean(sum_score - gold_score,dim = 0) # 负对数似然作为损失函数

    def _viterbi_decode(self, X):
        #   X   :  [batch_size,target_size]
        #  path :  [batch_size,seq_len    ]
        # score :  [batch_size            ]
        batch_size = X.size(0)
        seq_len = X.size(1)
        target_size = X.size(2)

        # 参考博客:https://www.cnblogs.com/pinard/p/7068574.html
        # 4. linear-CRF模型维特比算法流程
        psi = []              # ψ 用于回溯
        delta = [X[:,0,:]]    # δ 记录路径分数

        for i in range(seq_len-1):
            delta_prei = delta[i].unsqueeze(1).expand(batch_size,target_size,target_size)
            delta_i = X[:,i+1,:].unsqueeze(1).expand(batch_size,target_size,target_size)
            delta_i = delta_prei.transpose(1,2) + self.A.unsqueeze(0) + delta_i

            maxv,maxi = delta_i.max(dim = 1)
            delta.append(maxv)
            psi.append(maxi)

        path = []
        psi_end = delta[seq_len-1].argmax(dim = 1) # [batch_size]
        path.append(psi_end)
        score = delta[seq_len-1][np.arange(batch_size),psi_end]

        for i in range(seq_len-1):
            path.append(psi[seq_len-2-i][np.arange(batch_size),path[-1]])
            score += delta[seq_len-2-i][np.arange(batch_size),path[-1]]
        path.reverse()
        path = [p.unsqueeze(1) for p in path]
        path = torch.cat(path,dim = 1)
        return path,score

    def forward(self, X, labels_m,tags,chars):
        P = self._lstm_features(X,labels_m,tags,chars)

        return self._viterbi_decode(P)