# -*- coding = utf-8 -*-
# @Time : 2021/10/15 15:54
# @Author : Z_C_Lee
# @File : sampleToCSV.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import scipy.io as scio
from sampling.fisvdd import fisvdd
import collections
import os
from miVLAD_MNIST.Main import encode
from tools.processCSV import shuffle_data
import random


def extract(fileName, toFileName):
    """
    将提取的句子保存至CSV文件
    :param fileName: .CSV
    :param toFileName: .CSV格式
    :return:
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    file = pd.read_csv(fileName,header=None)

    all_data = scio.loadmat("data/sentence_vec/Amazon_train.mat")['data']
    label = []
    for i in (all_data[:, 1]).tolist():
        label.append(i[0, 0])
    dic = collections.Counter(label)
    label = np.array(label)

    flag = True
    save_data = None
    for key in dic:
        index = np.where(label == key)
        data = all_data[index,0].reshape(all_data[index,1].shape[1],1)
        # print(data)
        model = fisvdd(data, 0.7)
        sv_index_site = model.find_sv()
        sv_index = index[0][sv_index_site]
        # print(max(sv_index),np.array(file).shape)
        if flag:
            save_data = np.array(file)[sv_index]
            flag = False
        else:
            save_data = np.concatenate((save_data, np.array(file)[sv_index]))

    dataframe = pd.DataFrame({'a_name': save_data[:,0], 'b_name': save_data[:,1]})
    dataframe.to_csv(toFileName, index=False, sep=',',header=False)
    shuffle_data(toFileName)
    print("sample size:{}".format(save_data.shape[0]))
    print("--------end sampling-----------")


def extractByFolder_random(fileFolder, goalFolder):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    allFile = None

    for root, dirs, files in os.walk(fileFolder):
        allFile = files

    for file_name in allFile:
        ab_path = fileFolder + "\\" + file_name
        file = pd.read_csv(ab_path, header=None)

        fileName = ab_path.split("\\")[-1].split(".")[0]

        print(file[1].value_counts()[0] ,file[1].value_counts()[1])
        if file[1].value_counts()[0] > file[1].value_counts()[1]:
            key = 1
        else:
            key = 0
        differ = np.abs(file[1].value_counts()[0] - file[1].value_counts()[1])

        # flag = True
        # save_data = None
        index = np.where(np.array(file[1]) == key)

        sample_data = np.array(file)[index]
        save_data = []
        for i in range(300):
            random_index = np.random.randint(0, sample_data.shape[0])
            save_data.append(sample_data[random_index])
        save_data = np.array(save_data)
        save_data = np.concatenate((np.array(file), save_data))

        # if flag:
        #     save_data = np.array(file)[sv_index]
        #     flag = False
        # else:
        #     save_data = np.concatenate((save_data, np.array(file)[sv_index]))

        dataframe = pd.DataFrame({'a_name': save_data[:, 0], 'b_name': save_data[:, 1]})
        dataframe.to_csv(goalFolder + "/" + fileName + ".csv", index=False, sep=',', header=False)
        print("sample size:{}".format(sample_data.shape[0]))
    print("----------end sampling-----------")


def extractByFolder_fisvdd(fileFolder, goalFolder, flag):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    allFile=None

    for root, dirs, files in os.walk(fileFolder):
        allFile = files

    for file_name in allFile:
        ab_path=fileFolder+"/"+file_name
        file = pd.read_csv(ab_path, header=None)

        folderName = ab_path.split("/")[-3]
        fileName = ab_path.split("/")[-1].split(".")[0]

        dataFile = "data/sample_vec/" + folderName + "/" + fileName + ".mat"
        all_data = scio.loadmat(dataFile)['data']

        if flag:
            if file[1].value_counts()[0] > file[1].value_counts()[1]:
                key = 1
            else:
                key = 0
            differ = np.abs(file[1].value_counts()[0] - file[1].value_counts()[1])

            index = np.where(np.array(file[1]) == key)
            data = all_data[index, 0].reshape(all_data[index, 1].shape[1], 1)

            model = fisvdd(data, 0.6)
            sv_index_site = model.find_sv()
            sv_index = index[0][sv_index_site]
            sample_data = np.array(file)[sv_index]

            save_data = []
            for i in range(differ):
                random_index = np.random.randint(0, sample_data.shape[0])
                save_data.append(sample_data[random_index])

            save_data = np.array(save_data)
            save_data = np.concatenate((np.array(file),save_data))

        else:
            if file[1].value_counts()[0] < file[1].value_counts()[1]:
                key = 1
            else:
                key = 0
            # differ = np.abs(file[1].value_counts()[0] - file[1].value_counts()[1])

            index = np.where(np.array(file[1]) == key)
            reserve_index = np.where(np.array(file[1]) != key)

            data = all_data[index, 0].reshape(all_data[index, 1].shape[1], 1)
            reserve_data = np.array(file)[reserve_index]

            model = fisvdd(data, 0.6)
            sv_index_site = model.find_sv()
            sv_index = index[0][sv_index_site]
            sample_data = np.array(file)[sv_index]

            save_data = sample_data
            save_data = np.concatenate((reserve_data,save_data))

        dataframe = pd.DataFrame({'a_name': save_data[:, 0], 'b_name': save_data[:, 1]})
        dataframe.to_csv(goalFolder+"/"+fileName+".csv", index=False, sep=',', header=False)

        print("sample size:{}".format(sample_data.shape[0]))
    print("----------end sampling-----------")

def extractByFile_fisvdd(filename, goalFile, flag):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    file = pd.read_csv(filename, header=None)
    folderName, fileName = os.path.split(filename)
    folderName = folderName.split("/")[-3]
    name = fileName.split(".")[0]

    encode(filename)

    dataFile = "data/sample_vec/"+ folderName +"/" + name + ".mat"
    all_data = scio.loadmat(dataFile)['data']

    if flag:
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0

        index = np.where(np.array(file[1]) == key)
        reserve_index = np.where(np.array(file[1]) != key)

        data = all_data[index, 0].reshape(all_data[index, 1].shape[1], 1)
        reserve_data = np.array(file)[reserve_index]

        model = fisvdd(data, 0.35)
        sv_index_site = model.find_sv()
        sv_index = index[0][sv_index_site]
        sample_data = np.array(file)[sv_index]
        print("sample size:{}".format(sample_data.shape[0]))

        save_data = np.array(random.sample(sample_data.tolist(), file[1].value_counts()[0]))
        save_data = np.concatenate((reserve_data, save_data))

    else:
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0

        index = np.where(np.array(file[1]) == key)
        reserve_index = np.where(np.array(file[1]) != key)

        data = all_data[index, 0].reshape(all_data[index, 1].shape[1], 1)
        reserve_data = np.array(file)[reserve_index]

        model = fisvdd(data, 0.38)
        sv_index_site = model.find_sv()
        sv_index = index[0][sv_index_site]
        sample_data = np.array(file)[sv_index]
        print("sample size:{}".format(sample_data.shape[0]))

        save_data = np.array(random.sample(sample_data.tolist(), 10000))
        save_data = np.concatenate((reserve_data, save_data))

    dataframe = pd.DataFrame({'a_name': save_data[:, 0], 'b_name': save_data[:, 1]})
    dataframe.to_csv(goalFile, index=False, sep=',', header=False)
    shuffle_data(goalFile)

    print("----------end one file sampling-----------")


def extractByFile_random(filename, goalFile, flag):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    file = pd.read_csv(filename, header=None)

    if flag:
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0
        index = np.where(np.array(file[1]) == key)
        reserve_index = np.where(np.array(file[1]) != key)
        sample_data = np.array(file)[index]
        reserve_data = np.array(file)[reserve_index]
        save_data = np.array(random.sample(sample_data.tolist(), file[1].value_counts()[0]))
        print("sample size:{}".format(file[1].value_counts()[0]))
        save_data = np.concatenate((reserve_data,save_data))
        

    else:
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0
        differ = file[1].value_counts()[(key-1)*(-1)]
        index = np.where(np.array(file[1]) == key)
        reserve_index = np.where(np.array(file[1]) != key)

        sample_data = np.array(file)[index]
        reserve_data = np.array(file)[reserve_index]
        save_data = np.array(random.sample(sample_data.tolist(), 13000))
        print("sample size:{}".format(save_data.shape[0]))
        # for i in range(1000):
        #     random_index = np.random.randint(0, sample_data.shape[0])
        #     save_data.append(sample_data[random_index])
        # save_data = np.array(save_data)
        save_data = np.concatenate((reserve_data,save_data))

    dataframe = pd.DataFrame({'a_name': save_data[:, 0], 'b_name': save_data[:, 1]})
    dataframe.to_csv(goalFile, index=False, sep=',', header=False)
    shuffle_data(goalFile)
    print("----------end sampling-----------")


if __name__ == "__main__":
    goalFolder = r"D:\My_Code\Python\reduction\data\sample_files\Yelp"
    fileFolder = r"D:\My_Code\Python\reduction\data\initial_dataset\Yelp\train_files"
    testFolder = r"D:\My_Code\Python\reduction\data\initial_dataset\Yelp\test_files"
    # extractByFolder_fisvdd(fileFolder, goalFolder)
