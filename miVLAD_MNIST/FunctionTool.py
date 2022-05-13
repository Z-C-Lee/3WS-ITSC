"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0903, last modified in 2021 0511.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import scipy.io as scio
import gensim
import numpy as np
import torchtext.vocab as vocab
import re
import copy

# word2vec = gensim.models.KeyedVectors.load_word2vec_format("../nn/tools/GoogleNews-vectors-negative300.bin",
#                                                                binary=True,
#                                                                limit=200000)
cache_dir = "tools"
# glove = vocab.pretrained_aliases["glove.6B.300d"](cache=cache_dir)
glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)  # 与上面等价


def data_Cleaning(dataset):
    stop_word = get_stop_word("tools/baidu_stopwords.txt")
    for i in range(len(dataset)):
        t=0
        for j in range(len(dataset[i].review)):
            try:
                if(dataset[i].review[j-t] in stop_word):
                    dataset[i].review.remove(dataset[i].review[j - t])
                    t = t + 1
                    continue
                else:
                    var = glove[dataset[i].review[j-t]]
            except KeyError:
                dataset[i].review.remove(dataset[i].review[j-t])
                t=t+1
    return dataset


def dataCleaning(dataset):
    for i in range(len(dataset)):
        add = 0
        pre_data = copy.deepcopy(dataset[i].review)
        for j in range(len(pre_data)):
            try:
                glove.stoi[pre_data[j]]
            except KeyError:
                data_list = split_text(pre_data[j])
                dataset[i].review=np.delete(dataset[i].review,j+add)
                dataset[i].review = np.insert(dataset[i].review,j+add,data_list)
                dataset[i].review = dataset[i].review.tolist()
                add += len(data_list)-1
    return dataset


def get_stop_word(data_dir):
    with open(data_dir,'r',encoding='utf-8') as data:
        words=set()
        for line in data:
            line = line.strip().strip('\n')
            words.add(line)
    return words


def split_text(text):
    """
    切分文本，自定义一些规则
    :param text:
    :return: 分割序列下标
    """
    split_index=[]
    pattern1 = ";|\.|\?|\(|\)|,|'|!|:"
    for m in re.finditer(pattern1, text):
        idx=m.span()[0]
        if text[idx]=="'":
            split_index.append(idx)
        else:
            split_index.append(idx)
            split_index.append(idx+1)
    split_index = list(set([0, len(text)] + split_index))
    split_index = sorted(split_index)
    result_text = [text[split_index[i]:split_index[i+1]] for i in range(len(split_index)-1)]
    return result_text

def load_file(para_path, start, end):
    """
    Load file.
    :param
        para_file_name:
            The path of the given file.
    :return
        The data.
    """
    temp_type = para_path.split('.')[-1]

    if temp_type == 'mat':
        ret_data = scio.loadmat(para_path)
        return ret_data['data']

    elif temp_type == 'csv':
        from torchtext.legacy import data
        LABEL = data.LabelField()
        REVIEW = data.Field(lower=True)
        fields = [('review', REVIEW), ('label', LABEL)]

        # --------------------分  词-------------------
        train_data = data.TabularDataset(
            path=para_path,
            format='CSV',
            fields=fields,
            skip_header=False
        )
        train_data = dataCleaning(train_data)
        train_data = data_Cleaning(train_data)
        all_data = []

        for i in range(start, end):
            one_data = []
            data = []
            label = []
            if len(train_data[i].review)==0:
                data.append(np.array([0] * 300, dtype='float32'))
                data[0] = np.append(data[0], int(train_data[0].label))
            for j in range(len(train_data[i].review)):
                # try:
                data.append(glove[train_data[i].review[j]])
                # except KeyError:
                #     data.append(np.array([0] * 300, dtype='float32'))
                data[j] = np.append(data[j], int(train_data[i].label))

            data = np.array(data)
            one_data.append(data)
            label.append([int(train_data[i].label)])
            one_data.append(np.array(label))
            all_data.append(one_data)

        return np.array(all_data)


def print_progress_bar(para_idx, para_len):
    """
    Print the progress bar.
    :param
        para_idx:
            The current index.
        para_len:
            The loop length.
    """
    print('\r' + '▇' * int(para_idx // (para_len / 50)) + str(np.ceil((para_idx + 1) * 100 / para_len)) + '%', end='')


def mnist_bag_loader(train, mnist_path=None):
    """"""
    if mnist_path is None:
        mnist_path = "../Data"
    return DataLoader(datasets.MNIST(mnist_path,
                                     train=train,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])),
                                      batch_size=1,
                                      shuffle=False)


def get_k_cross_validation_index(num_x, k=10):
    """
    The get function.
    """
    rand_idx = np.random.permutation(num_x)
    temp_fold = int(np.floor(num_x / k))
    ret_tr_idx = []
    ret_te_idx = []
    for i in range(k):
        temp_tr_idx = rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx.append(temp_tr_idx)
        ret_te_idx.append(rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist())
    return ret_tr_idx, ret_te_idx
