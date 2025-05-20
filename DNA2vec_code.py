import numpy as np
import torch
import torch.nn as nn
from keras.layers.convolutional import Conv2D

def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []  # len(13774), 每行DNA长度972
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence.append(line.strip('\n'))

    k = 5
    kmer_list = []  #二维数组, len(13774,972-5+1=968)
    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            ind = index.index(sequence[number][i:i + k])
            seq.append(ind)
        kmer_list.append(seq)
    # kmer_list存的是每5个一单元 查找其在index的索引
    '''sum_length = 0
    cnt = 0
    for number in range(len(sequence)):
        sum_length += (len(sequence[number]) - k + 1)
        cnt = number
    average_length = round(sum_length / (cnt + 1))'''

    feature_word2vec = []
    for number in range(len(kmer_list)):
        #print(number)
        feature_seq = []  # 这个list长度应该是968*8=7744
        for i in range(len(kmer_list[number])):
            kmer_index = kmer_list[number][i]
            for j in word2vec[kmer_index].tolist():
                feature_seq.append(j)
        feature_seq_tensor = torch.Tensor(feature_seq)  # 转换为 PyTorch 一维张量
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)  # 变成了三维张量, shape(1,1,7744)
        feature_seq_tensor_avg = nn.AdaptiveAvgPool1d(1000 * 8)(feature_seq_tensor)  # 将其自适应平均池化, 变成了shape([1,1,8000])

        feature_seq_numpy = feature_seq_tensor_avg.numpy()
        feature_seq_numpy = np.squeeze(feature_seq_numpy)  # 移除 NumPy 数组中所有大小为 1 的维度
        feature_seq_numpy = np.squeeze(feature_seq_numpy)  # shape(8000,)
        feature_seq_list = feature_seq_numpy.tolist()  # 这个list长度8000

        feature_word2vec.append(feature_seq_list)

    return feature_word2vec  # 二维len(13774, 8000)


'''cell_lines = 'NHEK'
sets = 'train'
filename = 'data/' + cell_lines + '/' +sets + '/data.fasta'
# filename = 'data/' + cell_lines + '/' + element + '/test/test.fasta'
f = open('index_promoters.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')
word2vec = np.loadtxt('word2vec_promoters.txt')
feature_word2vec = word_embedding(filename, index, word2vec)
feature_word2vec = np.array(feature_word2vec)
print(feature_word2vec.shape)
np.savetxt('feature/' + cell_lines + '/' +sets + '/word2vec.txt', feature_word2vec)'''


cell_lines = 'K562'
filename = 'EPdata/' + cell_lines + '/data.fasta'
# filename = 'data/' + cell_lines + '/' + element + '/test/test.fasta'
f = open('index_promoters.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')  # 存为list数组类型, len(1024), 每个长度5
word2vec = np.loadtxt('word2vec_promoters.txt')  # shape(1024,8)
feature_word2vec = word_embedding(filename, index, word2vec)
feature_word2vec = np.array(feature_word2vec)  # shape(12774,8000)
print(feature_word2vec.shape)
np.savetxt('EPfeature/' + cell_lines + '/word2vec.txt', feature_word2vec)  # 存的都是启动子(5个碱基)分数

