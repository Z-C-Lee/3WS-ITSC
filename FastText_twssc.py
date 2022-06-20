# -*- coding = utf-8 -*-
# @Time : 2021/12/11 10:27
# @Author : Z_C_Lee
# @File : RNN_twssc.py
# @Software : PyCharm

import pandas as pd
import torch

from torchtext.legacy import data
from sampling.sampleToCSV import extractByFile_random, extractByFile_fisvdd
from tools.processCSV import find_lack_label
import numpy as np
import time
import os
from main.FastText.main import FastText, get_data_iter
from tools.processCSV import shuffle_data
import torch.optim as optim
import os
import logging
from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset
from torchtext.vocab import Vectors

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
class TWSSC:
    def __init__(self, trainFolder, validFolder, sampleFolder, all_train_filename, all_test_filename):
        self.alpha=None
        self.beta=None
        self.model=None
        self.TP=0
        self.FN=0
        self.FP=0
        self.TN=0
        self.current_lack_len=0
        self.current_all_correct=0
        self.current_lack_correct=0

        self.trainFolder=trainFolder
        self.validFolder=validFolder
        self.sampleFolder=sampleFolder

        self.all_train_filename=all_train_filename
        self.all_test_filename=all_test_filename

        self.train_file_all_num=None

    def split_valid(self, filename):
        head, tail = os.path.split(filename)

        file=pd.read_csv(filename, header=None)
        train = file.sample(frac=0.9, random_state=1, axis=0)
        valid = file[~file.index.isin(train.index)]

        train = train.sort_values(by=1, ascending=False)
        train.to_csv('{0}{1}'.format(head+"/main_train/",tail), header=False, encoding="utf-8", sep=",",
                     index=False)
        valid.to_csv('{0}{1}'.format(head+"/valid_files/",tail), header=False, encoding="utf-8", sep=",",
                     index=False)

    def compute_lamb(self, num_file):
        """
        计算代价矩阵
        :param num_file:
        :return:
        """
        # up = 20 / self.train_file_all_num
        # lamb_PN = 60
        # lamb_NP = 40
        # lamb_BP = num_file * up
        # lamb_BN = lamb_BP

        # up = 10 / self.train_file_all_num
        # lamb_PN = 60 - (num_file - 1) * up
        # lamb_NP = 40 - (num_file - 1) * up
        # lamb_BP = num_file * up
        # lamb_BN = lamb_BP

        self.lamb_PN = 750
        self.lamb_NP = 150
        self.lamb_BP = 40
        self.lamb_BN = 40

        return 0,  self.lamb_PN,  self.lamb_BP,  self.lamb_BN,  self.lamb_NP, 0

    def compute_threshold(self, num_file):
        """
        计算阈值
        :param all_file_len:
        :param num_file:
        :return:
        """
        lamb_PP, lamb_PN, lamb_BP, lamb_BN, lamb_NP, lamb_NN = self.compute_lamb(num_file)
        self.alpha = (lamb_PN - lamb_BN) / ((lamb_PN - lamb_BN) + (lamb_BP - lamb_PP))
        self.beta = (lamb_BN - lamb_NN) / ((lamb_BN - lamb_NN) + (lamb_NP - lamb_BP))

    def compute_entropy(self, prob_arr):
        value = 0
        for prob in prob_arr:
            value += -prob * np.log2(prob)
        return value

    def train_model(self, net, train_iter, valid_iter, epoch, lr, batch_size):
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("begin training")
        net.train()  # 必备，将模型设置为训练模式
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        all_loss = 0
        all_len = 0
        for i in range(epoch):  # 多批次循环
            for batch_idx, batch in enumerate(train_iter):
                target = batch.label.to(DEVICE)
                data = batch.review.to(DEVICE)
                optimizer.zero_grad()  # 清除所有优化的梯度
                output = net(data)  # 传入数据并前向传播获取输出
                loss = criterion(output, target)
                all_loss += loss.item()
                all_len += 1
                loss.backward()
                optimizer.step()

                # 打印状态信息
            print("train epoch=" + str(i) + ",loss=" + str(all_loss / (batch_size * all_len)))
        print('Finished Training')

        net.eval()  # 必备，将模型设置为训练模式
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_iter):
                # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
                target = batch.label.to(DEVICE)
                data = batch.review.to(DEVICE)
                logging.info("test batch_id=" + str(i))
                outputs = net(data)
                # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
                _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
                total += target.size(0)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
            print('Accuracy of the network on test set: %d %%' % (100 * correct / total))
                # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
                # logging.info("test_acc=" + str(test_acc))

    def one_file_test(self, test_iter, lack, file_num, flag=True):
        """
        测试模型
        :param model: 模型
        :param criterion: 损失函数
        :param test_iter: 测试集迭代器
        :param lack: 待抽样标签
        :param file_num: 文件编号
        :param flag: 是否为最后一个文件
        :return:
        """

        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("-------one file test----------")
        self.compute_threshold(file_num)

        self.model.eval()
        next_test_index = []

        current_lack_correct = 0
        current_all_correct = 0
        current_lack_len = 0
        current_all_len = 0

        lack_correct_ratio = 0
        ample_correct_ratio = 0

        num_1 = 0
        num_0 = 0

        with torch.no_grad():  # 不进行梯度计算
            if flag:
                for index, batch in enumerate(test_iter):
                    target = batch.label.to(DEVICE)
                    data = batch.review.to(DEVICE)
                    outputs = self.model(data)
                    batch_prob = torch.softmax(outputs, dim=1)

                    # -----------------概率三支-------------------------
                    # for prob in batch_prob:
                    #     all_entropy.append(prob[0])
                    #
                    # all_entropy = np.array(all_entropy)
                    # flags1 = np.argwhere((all_entropy >= 0.9) | (all_entropy <= 0.15)).squeeze().tolist()
                    # flags2 = (np.argwhere((0.15 < all_entropy) & (all_entropy < 0.9)).squeeze() + (len(batch) * index)).tolist()
                    # -------------------------------------------------

                    # -------------------三支代价敏感---------------------
                    if ((self.alpha > batch_prob[0][0]) and (batch_prob[0][0] > self.beta)):
                        next_test_index.append(index)
                        if batch.label[0].item() != lack:
                            num_1 += 1
                        else:
                            num_0 += 1
                    else:
                        current_all_len += 1
                        predict_label = batch_prob.argmax(1)[0].item()
                        if ((target[0].item() == lack) and (predict_label == lack)):
                            self.current_lack_correct += 1
                            current_lack_correct += 1

                        if (predict_label == target[0].item()):
                            self.current_all_correct += 1
                            current_all_correct += 1

                        if (target[0].item() == lack):
                            self.current_lack_len += 1
                            current_lack_len += 1
                    # --------------------------------------------------
            else:
                print("-------------final file---------------")
                for index, batch in enumerate(test_iter):

                    # batch.label = (batch.label - 1) * (-1)
                    target = batch.label.to(DEVICE)
                    data = batch.review.to(DEVICE)
                    outputs = self.model(data)
                    prob = torch.softmax(outputs, dim=1)

                    if prob[0][0] < 0.5:
                        predict_label = 1
                    else:
                        predict_label = 0

                    if target[0].item() != lack:
                        num_1 += 1
                    else:
                        num_0 += 1

                    current_all_len += 1
                    if ((target.item() == lack) and (predict_label == lack)):
                        self.current_lack_correct += 1
                        current_lack_correct += 1

                    if (predict_label == target[0].item()):
                        self.current_all_correct += 1
                        current_all_correct += 1

                    if (target[0].item() == lack):
                        self.current_lack_len += 1
                        current_lack_len += 1

            if ((current_lack_len) != 0):
                lack_correct_ratio = current_lack_correct / current_lack_len
                print(current_lack_correct, current_lack_len)
            else:
                print("---No Lack Data---")

            if ((current_all_len - current_lack_len) != 0):
                ample_correct_ratio = (current_all_correct - current_lack_correct) / (
                            current_all_len - current_lack_len)
                print((current_all_correct - current_lack_correct), (current_all_len - current_lack_len))
            else:
                print("---No Ample Data---")

            self.TN += current_lack_correct
            self.FP += (current_lack_len - current_lack_correct)
            self.TP += current_all_correct - current_lack_correct
            self.FN += current_all_len - current_lack_len - (current_all_correct - current_lack_correct)

            print("next 0 and 1:{} {}".format(num_0, num_1))
            print("now 0 and 1:{} {}".format(current_lack_len, current_all_len - current_lack_len))

            # test_loss /= len(test_iter)
            # print("Test_Loss:{}".format(test_loss))
            print("Lack_Acc:{:.4f}".format(lack_correct_ratio))
            print("Ample_Acc:{:.4f}\n".format(ample_correct_ratio))

            return next_test_index

    def process_data(self, data, REVIEW):
        for i in range(len(data)):
            one_data = data[i].review
            for j in range(len(one_data)):
                one_data[j] = REVIEW.vocab.stoi[one_data[j]]
            if len(one_data) % 3 != 0:
                add = [1] * (3 - len(one_data) % 3)
                one_data = one_data + add
            one_data = np.array(one_data).reshape(int(len(one_data) / 3), 3)
            data[i].review =  one_data.tolist()

    def compute_F1(self):
        Precision = self.TP / (self.TP + self.FP)
        Recall = self.TP / (self.TP + self.FN)
        F1 =  2 * (Precision*Recall) / (Precision + Recall)
        print("F1:{:.4f}".format(F1))


    def main(self, num_time):
        lack_label = find_lack_label(self.all_train_filename)
        # --------------------------------------------------------------------------------

        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        word2vec_dir = "tools/glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
        sentence_max_size = 50  # 每篇文章的最大词数量
        batch_size = 64
        epoch = 5  # 迭代次数
        emb_dim = 300  # 词向量维度
        lr = 0.0001
        hidden_size = 100
        label_size = 2

        # 设置表头
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LABEL = data.Field(sequential=False, use_vocab=False)
        REVIEW = data.Field(sequential=True, lower=True, batch_first=True)
        fields = [('review', REVIEW), ('label', LABEL)]
        all = TabularDataset(path=self.all_train_filename, format="csv", fields=fields, skip_header=True)
        vectors = Vectors(name=word2vec_dir)
        REVIEW.build_vocab(all, vectors=vectors)
        LABEL.build_vocab(all)
        vocab = REVIEW.vocab

        for root, dirs, files in os.walk(self.trainFolder, topdown=False):
            train_files = files

        self.train_file_all_num = len(train_files)

        for k in range(num_time):
            start_time = time.time()
            print("----------第{}次----------".format(k))

            # self.model = LSTM(hidden_size=100, embedding_dim=300, vocab_size=20002, out_size=2)
            self.model = FastText(vocab=vocab, vec_dim=emb_dim, label_size=label_size, hidden_size=hidden_size)
            self.model=self.model.to(DEVICE)
            self.TN = 0
            self.FP = 0
            self.TP = 0
            self.FN = 0
            self.current_lack_correct = 0
            self.current_all_correct = 0
            self.current_lack_len = 0

            index = []
            flag = True
            file_num = 0
            for trainFile in train_files:
                print("file name: {}".format(trainFile))
                file_num += 1
                if trainFile == train_files[-1]:
                    flag = False

                # self.split_valid(self.trainFolder + "/" + trainFile)
                # extractByFile_fisvdd(self.trainFolder + "/main_train/" + trainFile, self.sampleFolder + "/" + trainFile, True)
                extractByFile_random(self.trainFolder + "/main_train/" + trainFile, self.sampleFolder + "/" + trainFile, False)

                if (index != None and len(index) != 0):
                    main_test_file = pd.read_csv("data/my_test.csv", header=None, encoding="utf-8")
                    main_test_file = np.array(main_test_file)[np.array(index)]
                elif file_num == 1:
                    one_test_file = pd.read_csv(self.all_test_filename, header=None, encoding="utf-8")
                    main_test_file = np.array(one_test_file)
                else:
                    break
                dataframe = pd.DataFrame({'a_name': main_test_file[:, 0], 'b_name': main_test_file[:, 1]})
                dataframe.to_csv("data/my_test.csv", header=False, encoding="utf-8", sep=",", index=False)

                file = pd.read_csv("data/my_test.csv", header=None,
                                             encoding="utf-8")

                train_iter, test_iter, valid_iter = get_data_iter(fields,
                                                            self.sampleFolder + "/" + trainFile,
                                                             "data/my_test.csv",
                                                             self.trainFolder + "/valid_files/" + trainFile,
                                                             batch_size)

                # 文本批处理
                self.train_model(self.model, train_iter, valid_iter, epoch, lr, batch_size)

                index = self.one_file_test(test_iter, lack_label, file_num, flag)

            lack_arr = self.TN / (self.TN + self.FP)
            ample_arr = self.TP / (self.TP + self.FN)
            my_correct_arr = (self.TN + self.TP) / (self.TN + self.FP + self.TP + self.FN)

            print("--------------all files test----------------")
            print(self.TP, self.FP, self.TN, self.FN)
            print(self.current_lack_len, self.current_all_correct, self.current_lack_correct)
            print("Lack_Accuracy:{:.4f}".format(lack_arr))
            print("Ample_Accuracy:{:.4f}".format(ample_arr))
            print("Accuracy:{:.4f}".format(my_correct_arr))
            TC = self.lamb_PN * self.FP + self.lamb_NP * self.FN
            print("AC:{:.4f}".format(TC / (self.TN + self.FP + self.TP + self.FN)))
            self.compute_F1()

            end_time = time.time()
            d_time = end_time - start_time

            print("程序运行时间：%.8s s" % d_time)  # 显示到微秒

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

if __name__ == '__main__':
    sampleFolder = "data/sample_files/Amazon"
    trainFolder = "data/initial_dataset/Amazon/train_files"
    validFolder = "data/initial_dataset/Amazon/train_files/valid_files"
    all_train_filename="data/initial_dataset/Amazon/Amazon_Train.csv"
    all_test_filename="data/initial_dataset/Amazon/Amazon_Test1.csv"
    sys.stdout = Logger(stream=sys.stdout)

    twssc = TWSSC(trainFolder=trainFolder, validFolder=validFolder, sampleFolder=sampleFolder, all_train_filename=all_train_filename, all_test_filename=all_test_filename)
    twssc.main(10)