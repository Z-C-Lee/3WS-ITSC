# -*- coding = utf-8 -*-
# @Time : 2022/1/13 15:51
# @Author : Z_C_Lee
# @File : FastText.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FastText(nn.Module):
    def __init__(self, hidden_size,embed_size,vocab_size,num_classes):
        """
        parameter configuration
        :param hidden_size: Hidden layer size
        :param embed_size: Embedding layer size
        :param vocab_size: vocab size
        :param num_classes: Number of classes
        """
        super(FastText, self).__init__()
        self.embedding_word = nn.Embedding(vocab_size, embed_size,
                                           padding_idx=vocab_size - 1)
        self.embedding_bigram = nn.Embedding(vocab_size, embed_size,
                                             padding_idx=vocab_size - 1)
        self.embedding_trigram = nn.Embedding(vocab_size, embed_size,
                                              padding_idx=vocab_size - 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        forward propagation
        :param x: input text data, shape=[batch_size,seq_length],len(list)=3
        :return: Classification of sentences [batch, num_classes]
        """
        embed_bow = self.embedding_word(x[0])
        embed_bigram = self.embedding_bigram(x[1])
        embed_trigram = self.embedding_trigram(x[2])

        # concatenate the results
        out = torch.cat((embed_bow, embed_bigram, embed_trigram), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out