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

def extractByFile_fisvdd(filename, goalFile, flag):
    """
    Three samples by file name
    :param filename: input file
    :param goalFile: output file
    :param flag: True(specify sample size), False(custom sample size)
    :return:
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    file = pd.read_csv(filename, header=None)
    folderName, fileName = os.path.split(filename)
    folderName = folderName.split("/")[-3]
    name = fileName.split(".")[0]

    encode(filename)

    dataFile = "data/sample_vec/"+ folderName +"/" + name + ".mat"
    all_data = scio.loadmat(dataFile)['data']

    if flag:
        # Sampling label confirmation
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0
        # Get the sample number of the majority class in the file
        index = np.where(np.array(file[1]) == key)
        # Get the sample number of the minority class in the file
        reserve_index = np.where(np.array(file[1]) != key)
        # Get the sample of the majority class in the file
        data = all_data[index, 0].reshape(all_data[index, 1].shape[1], 1)
        # Get the sample of the minority class in the file
        reserve_data = np.array(file)[reserve_index]
        # fisvdd model definition
        model = fisvdd(data, 0.35)
        # Get support vector number
        sv_index_site = model.find_sv()
        sv_index = index[0][sv_index_site]
        sample_data = np.array(file)[sv_index]
        print("sample size:{}".format(sample_data.shape[0]))
        # Integrate training samples
        save_data = np.array(random.sample(sample_data.tolist(), file[1].value_counts()[0]))
        save_data = np.concatenate((reserve_data, save_data))

    else:
        # Explain the same as "if"
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
    """
    Randomly undersample based on filename
    :param filename: input file
    :param goalFile: output file
    :param flag: True(specify sample size), False(custom sample size)
    :return:
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    file = pd.read_csv(filename, header=None)

    if flag:
        # Sampling label confirmation
        if file[1].value_counts()[0] < file[1].value_counts()[1]:
            key = 1
        else:
            key = 0
        # Get the sample number of the majority class in the file
        index = np.where(np.array(file[1]) == key)
        # Get the sample number of the minority class in the file
        reserve_index = np.where(np.array(file[1]) != key)
        # Get the sample of the majority class in the file
        sample_data = np.array(file)[index]
        # Get the sample of the minority class in the file
        reserve_data = np.array(file)[reserve_index]
        # Integrate training samples
        save_data = np.array(random.sample(sample_data.tolist(), file[1].value_counts()[0]))
        print("sample size:{}".format(file[1].value_counts()[0]))
        save_data = np.concatenate((reserve_data,save_data))

    else:
        # Explain the same as "if"
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
