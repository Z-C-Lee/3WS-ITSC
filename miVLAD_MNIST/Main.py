"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0907 1601, last modified in 2021 0415.
"""

import warnings
import time
from miVLAD_MNIST.miVLAD import miVLAD
import scipy.io as scio
import numpy as np
import pandas as pd
import collections


warnings.filterwarnings('ignore')
def encode(file_name):
    """
    """
     # "D:/Data/OneDrive/文档/Code/MIL1/Data/Text/talk_religion_misc.mat"

    print("=================================================")
    print("BAMIC with %s" % file_name.split("/")[-1].split(".")[0])

    file = pd.read_csv(file_name,header=None)
    all_data = np.array(file)

    label = all_data[:, 1]
    dic = collections.Counter(label)
    label = np.array(label)

    start = 0
    flag=False
    for key in dic:
        index = np.where(label == key)
        end = max(index[0])

        mil = miVLAD(start, end + 1, file_name, k_m=1)
        start = end + 1
        data_iter = mil.get_mapping()

        folderName = file_name.split("/")[-4]
        fileName = file_name.split("/")[-1].split(".")[0]

        dataFile = "data/sample_vec/"+folderName+"/"+fileName+".mat"
        if flag:
            data = scio.loadmat(dataFile)
            data = data['data']
            data = np.vstack((data, data_iter))
            scio.savemat(dataFile, {'data': data})
        else:
            scio.savemat(dataFile, {'data': data_iter})

        flag=True

    # total = len(open(file_name,'rb').readlines())
    # time = total // 5000 + 1
    # remain = total % 5000
    #
    # for i in range(time):
    #     start = i * 5000
    #     if i==time-1:
    #         if remain == 0:
    #             break
    #         end = i*5000 + remain
    #         mil = miVLAD(start, end, file_name, k_m=1)
    #         data_iter = mil.get_mapping()
    #     else:
    #         end = (i+1) * 5000
    #         mil = miVLAD(start, end, file_name, k_m=1)
    #         data_iter = mil.get_mapping()
    #
    #     dataFile = r'D:\My_Code\Python\reduction\data\sample_vec\IMDB_Train.mat'
    #     try:
    #         data = scio.loadmat(dataFile)
    #         data = data['data']
    #         data = np.vstack((data, data_iter))
    #         scio.savemat(dataFile, {'data': data})
    #     except:
    #         scio.savemat(dataFile, {'data': data_iter})


if __name__ == '__main__':
    import os
    file_dir = r"D:\My_Code\Python\reduction\data\initial_dataset\Amazon\train_files"
    neg_file = None
    for root, dirs, files in os.walk(file_dir):
        neg_file = files

    s_t = time.time()
    for file_name in neg_file:
        encode(file_dir+ "\\" + file_name)
    print("%.4f" % (time.time() - s_t))
