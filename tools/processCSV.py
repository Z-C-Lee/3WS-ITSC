# -*- coding = utf-8 -*-
# @Time : 2021/11/13 18:16
# @Author : Z_C_Lee
# @File : processCSV.py
# @Software : PyCharm

import pandas as pd
import os
from sklearn.utils import shuffle
import numpy as np

def shuffle_data(filename):
    """
    rearrange data
    :param filename: input file name
    :return:
    """
    data = pd.read_csv(filename,header=None)
    data = shuffle(data)
    data.to_csv(filename, index=False, header=False)

def sort_data(filename):
    """
    Sort by data labels
    :param filename: input file name
    :return:
    """
    data = pd.read_csv(filename,header=None,encoding="utf-8", sep=",")
    data = data.sort_values(by=1, ascending=True)
    data.to_csv(filename, index=False, header=False)

def CSV_info(filename):
    """
    Printing basic information of CSV file
    :param filename: input file name
    :return:
    """
    df = pd.read_csv(filename,header=None)
    print(df[1].value_counts())

def del_sample(filename):
    """
    delete some samples
    :param filename: input file name
    :return:
    """
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    data = data.tolist()
    data = data[12500:]
    data = np.array(data)
    dataframe = pd.DataFrame({'a_name': data[:, 0], 'b_name': data[:, 1]})
    dataframe.to_csv(filename, header=False, encoding="utf-8", sep=",", index=False)

def find_lack_label(file_name):
    """
    Find minority class sample labels
    :param file_name: input file name
    :return:
    """
    df = pd.read_csv(file_name, header=None)
    if df[1].value_counts()[0] > df[1].value_counts()[1]:
        return 1
    else:
        return 0

if __name__ == '__main__':
    filename="data\sample_files\Yelp\0Yelp_Train.csv"
    # shuffle_data(filename)
    # get_label_data(filename)
    # del_sample(filename)
    CSV_info(filename)
    # sort_data(filename)