"""
@author: Inki
@contact: inki.yinji@gamil.com
@version: Created in 2020 1123, last modified in 2020 1123.
@note: You can refer this blog of https://blog.csdn.net/weixin_44575152/article/details/106600849
"""

import numpy as np
from miVLAD_MNIST.Prototype import MIL
from sklearn.cluster import MiniBatchKMeans
from miVLAD_MNIST.I2I import dis_euclidean
from miVLAD_MNIST.FunctionTool import get_k_cross_validation_index


class miVLAD(MIL):
    """
    The algorithm of milVLAD.
    @param:
        k:
            The times of k-th cross validation.
        k_m:
            The clustering center numbers fro kMeans.
    @attribute:
        centers:
            The clustering centers.
    """

    def __init__(self, start, end, path, k=10, k_m=1, bag_space=None):
        super(miVLAD, self).__init__(path, start, end, bag_space=bag_space)
        self.k = k
        self.k_m = k_m
        self.tr_idx = []
        self.te_idx = []

    def __bag_mapping(self, idx, centers, labels):
        """
        Mapping each given bag by using centers.
        @param:
            idx:
                The index of bag to be mapped.
            centers:
                The clustering centers of kMeans clustering algorithm.
            labels:
                The label for bag's instances which indicate the center to which the instance belongs.
        """
        ret_vec = np.zeros((self.k_m, self.num_att))
        idx_ins = 0
        for ins in self.bag_space[idx][0][:, :-1]:
            ret_vec[labels[idx_ins]] += ins - centers[labels[idx_ins]]
            idx_ins += 1
        ret_vec = np.resize(ret_vec, self.k_m * self.num_att)
        ret_vec = np.sign(ret_vec) * np.sqrt(np.abs(ret_vec))
        return ret_vec / dis_euclidean(ret_vec, np.zeros_like(ret_vec))

    def get_mapping(self):
        """
        Mapping bags to vectors.
        """
        all_data = []
        all_label = []
        temp_tr_ins_idx = [0]
        for tr_idx in range(self.num_bag):
            for ins in self.bag_space[tr_idx][0][:, :-1]:
                all_data.append(ins)
            all_label.append(self.bag_space[tr_idx][1][0,0])
            temp_tr_ins_idx.append(self.bag_size[tr_idx] + temp_tr_ins_idx[-1])
        all_data = np.array(all_data)
        temp_kmeans = MiniBatchKMeans(self.k_m)
        temp_kmeans.fit(all_data)
        temp_centers = temp_kmeans.cluster_centers_
        temp_labels = temp_kmeans.labels_

        re_data = []
        # ret_tr_vec = np.zeros((self.num_bag, self.k_m * self.num_att))
        for idx_bag in range(self.num_bag):
            re_data.append([])
            # ret_tr_vec[idx_bag] = self.__bag_mapping(idx_bag, temp_centers,
            #                                          temp_labels[temp_tr_ins_idx[idx_bag]:
            #                                                      temp_tr_ins_idx[idx_bag + 1]])
            re_data[idx_bag].append(self.__bag_mapping(idx_bag, temp_centers,
                                                     temp_labels[temp_tr_ins_idx[idx_bag]:
                                                                 temp_tr_ins_idx[idx_bag + 1]]))
            re_data[idx_bag].append(np.array([all_label[idx_bag]]))


        return re_data

        # self.tr_idx, self.te_idx = get_k_cross_validation_index(self.num_bag)
        # for loop_k in range(self.k):
        #     # Step 1. Clustering to k_m blocks.
        #     temp_tr_ins = []
        #     temp_tr_ins_idx = [0]
        #     temp_tr_idx = self.tr_idx[loop_k]
        #
        #     for tr_idx in temp_tr_idx:
        #         for ins in self.bag_space[tr_idx][0][:, :-1]:
        #             temp_tr_ins.append(ins)
        #         temp_tr_ins_idx.append(self.bag_size[tr_idx] + temp_tr_ins_idx[-1])
        #
        #     temp_tr_ins = np.array(temp_tr_ins)
        #
        #     temp_kmeans = MiniBatchKMeans(self.k_m)
        #     temp_kmeans.fit(temp_tr_ins)
        #     temp_centers = temp_kmeans.cluster_centers_
        #     temp_labels = temp_kmeans.labels_
        #
        #     # Step 2. Mapping.
        #     temp_num_tr = len(temp_tr_idx)
        #     ret_tr_vec = np.zeros((temp_num_tr, self.k_m * self.num_att))
        #     for idx_bag in range(temp_num_tr):
        #         ret_tr_vec[idx_bag] = self.__bag_mapping(temp_tr_idx[idx_bag], temp_centers,
        #                                                  temp_labels[temp_tr_ins_idx[idx_bag]:
        #                                                              temp_tr_ins_idx[idx_bag + 1]])
        #     temp_te_idx = self.te_idx[loop_k]
        #     temp_num_te = len(temp_te_idx)
        #
        #     ret_te_vec = np.zeros((temp_num_te, self.k_m * self.num_att))
        #     for idx_bag in range(temp_num_te):
        #         temp_labels = []
        #         for ins in self.bag_space[temp_te_idx[idx_bag]][0][:, :self.num_att]:
        #             temp_dis = []
        #             for center in temp_centers:
        #                 temp_dis.append(dis_euclidean(ins, center))
        #             temp_sorted_dis_idx = np.argsort(temp_dis)
        #             temp_labels.append(temp_sorted_dis_idx[0])
        #         ret_te_vec[idx_bag] = self.__bag_mapping(temp_te_idx[idx_bag], temp_centers, temp_labels)
        #     yield ret_tr_vec, self.bag_lab[temp_tr_idx], ret_te_vec, self.bag_lab[temp_te_idx], None
