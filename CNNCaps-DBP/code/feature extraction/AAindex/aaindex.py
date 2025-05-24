import pandas as pd
import numpy as np


def AAindex(sequence):
    obj = pd.read_csv('./AAindex_12.csv')
    pro_name_list = obj['AccNo'].tolist()

    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    AAindex_dict = {}
    for ele in AA_list_sort:
        AAindex_dict[ele] = obj[ele].tolist()
    AAindex_dict['X'] = [0] * 12  # 用于填充的特征向量
    feature_vector = []
    for item in sequence:
        feature_vector.extend(AAindex_dict[item])
    return feature_vector


def pad_or_truncate_sequence(sequence, fixed_length=1000):
    if len(sequence) < fixed_length:
        sequence += 'X' * (fixed_length - len(sequence))
    else:
        sequence = sequence[:fixed_length]
    return sequence


data = pd.read_csv("tst.csv")
data.columns = ['Sequences', 'Label']


def feature_extract(data):
    train = data.Sequences.values
    batch_size = len(train)
    fixed_length = 500
    embedding = 12
    feature_matrix = np.zeros((batch_size, fixed_length, embedding))
    for i, seq in enumerate(train):
        # 固定序列长度为1000
        seq = pad_or_truncate_sequence(seq, fixed_length)
        seq_features = AAindex(seq)
        seq_features = np.array(seq_features).reshape((fixed_length, embedding))
        feature_matrix[i] = seq_features
    return feature_matrix


feature_matrix = feature_extract(data)

# 仅保存特征矩阵
np.save('test_AAindex.npy', feature_matrix)
