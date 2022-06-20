# -*- coding = utf-8 -*-
# @Time : 2021/12/12 16:16
# @Author : Z_C_Lee
# @File : TextRNN.py
# @Software : PyCharm

import torch

class TextRNN(torch.nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, out_size):
        """
        parameter configuration
        :param hidden_size: Hidden layer size
        :param embedding_dim: Embedding layer size
        :param vocab_size: vocab size
        :param out_size: Number of classes
        """
        super(TextRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.encoder = torch.nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     dropout=0.5,
                                     num_layers=1)
        self.dropout = torch.nn.Dropout(0.5)
        self.predictor = torch.nn.Linear(hidden_size, out_size)

    def forward(self, seq):
        """
        forward propagation
        :param x: input text data
        :return: Classification of sentences
        """
        # embedding encoding
        code = self.embedding(seq)
        output, (hidden, cell) = self.encoder(code)
        hidden = self.dropout(hidden)
        predict = self.predictor(hidden.squeeze(0))
        return predict

# class TextRNN(torch.nn.Module):
#     def __init__(self, hidden_size, embedding_dim, vocab_size, out_size):
#         super(TextRNN, self).__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
#         self.encoder = torch.nn.LSTM(input_size=embedding_dim,
#                                      hidden_size=hidden_size,
#                                      bidirectional=True,
#                                      dropout=0.5,
#                                      num_layers=1)
#         self.predictor = torch.nn.Linear(2*hidden_size, out_size)
#         self.dropout = torch.nn.Dropout(0.5)
#
#     def forward(self, seq):
#         embedded = self.dropout(self.embedding(seq))
#         output, (hidden, cell) = self.encoder(embedded)
#         # output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
#         hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
#         predict = self.predictor(hidden)
#         return predict
