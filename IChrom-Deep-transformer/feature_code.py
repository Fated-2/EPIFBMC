import numpy as np

def one_hot(sequence):
    num = len(sequence)
    length = len(sequence[0])
    feature = np.zeros((num, length, 4))
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
