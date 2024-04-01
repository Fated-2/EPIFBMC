import numpy as np

def one_hot(sequence):
    """
    将DNA序列数据中的每个字符(A T C G)转换成长度为4的向量

    :param sequence: 要处理的序列数据
    :return: 返回填充好的三维数组feature
    """
    num = len(sequence)  # 表示一维数组的长度
    length = len(sequence[0])  # 表示一维数组中的第一个元素(即字符串)的长度
    feature = np.zeros((num, length, 4))  # 一个三维的数组
    for i in range(num):
        for j in range(length):
            if sequence[i][j] == 'A':
                feature[i, j] = [1, 0, 0, 0]
            elif sequence[i][j] == 'T':
                feature[i, j] = [0, 1, 0, 0]
            elif sequence[i][j] == 'C':
                feature[i, j] = [0, 0, 1, 0]
            elif sequence[i][j] == 'G':
                feature[i, j] = [0, 0, 0, 1]
    return feature
