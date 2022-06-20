# -*- coding = utf-8 -*-
# @Time : 2022/1/13 15:44
# @Author : Z_C_Lee
# @File : TextCNN.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, out_size):
        """
        parameter configuration
        :param hidden_size: Hidden layer size
        :param embedding_dim: Embedding layer size
        :param vocab_size: vocab size
        :param out_size: Number of classes
        """
        super(TextCNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Sequential(
                            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=5),
                            nn.ReLU())
        self.f1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """
        forward propagation
        :param x: input text data
        :return: Classification of sentences
        """
        # batch_size,length,embedding_size  64*600*64
        x = self.embedding(x)
        # Transpose the dimension of tensor
        x = x.permute(1, 2, 0)
        self.max_text_len = x.shape[2]
        # 16*100*len-4 after Conv1 operation, unchanged after ReLU, 16*100*1 after MaxPool1d
        x = self.conv(x)
        x = F.avg_pool1d(x, kernel_size=self.max_text_len - 5 + 1)
        # 64*256
        x = x.view(-1, x.size(1))
        x = F.dropout(x, 0.5)
        # 64*10 batch_size * class_num
        x = self.f1(x)
        return x
