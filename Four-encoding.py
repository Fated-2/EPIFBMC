import numpy as np
import collections
import pdb
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical


# KNF编码
def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  # {'A':0, 'C':1, 'G':2, 'T':3}
    return word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base   # 只保留整数部分
        ch1 = chars[n % base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index   #关于'A G C T'其中3个元素的16个全排列


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq, kmer, coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer.replace('T', 'T')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

'''
def coden(seq, kmer, tris):
    coden_dict = tris
    freq_dict = frequency(seq, kmer, coden_dict)
    vectors = np.zeros((5000, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i + kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'U')]] = value / 100
    return vectors
'''
def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((5000, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'T')]] = 1
    return vectors.tolist()

'''
由于3-mers有64种组合，只有21种不同的符号(20个氨基酸加一个停止密码子)，每个氨基酸可能对应多个密码子。
该方法不仅降低了经典k-mer方法的特征维数，而且将具有共同生物学性质的3-mer进行了分组。
'''
coden_dict1 = {'GCT': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
              'TGT': 1, 'TGC': 1,  # systeine<C>
              'GAT': 2, 'GAC': 2,  # aspartic acid<D>
              'GAA': 3, 'GAG': 3,  # glutamic acid<E>
              'TTT': 4, 'TTC': 4,  # phenylanaline<F>
              'GGT': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
              'CAT': 6, 'CAC': 6,  # histidine<H>
              'ATT': 7, 'ATC': 7, 'ATA': 7,  # isoleucine<I>
              'AAA': 8, 'AAG': 8,  # lycine<K>
              'TTA': 9, 'TTG': 9, 'CTT': 9, 'CTC': 9, 'CTA': 9, 'CTG': 9,  # leucine<L>
              'ATG': 10,  # methionine<M>
              'AAT': 11, 'AAC': 11,  # asparagine<N>
              'CCT': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
              'CAA': 13, 'CAG': 13,  # glutamine<Q>
              'CGT': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
              'TCT': 15, 'TCC': 15, 'TCA': 15, 'TCG': 15, 'AGT': 15, 'AGC': 15,  # serine<S>
              'ACT': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
              'GTT': 17, 'GTC': 17, 'GTA': 17, 'GTG': 17,  # valine<V>
              'TGG': 18,  # tryptophan<W>
              'TAT': 19, 'TAC': 19,  # tyrosine(Y)
              'TAA': 20, 'TAG': 20, 'TGA': 20,  # STOP code
              }

def coden1(seq):
    vectors = np.zeros((5000, 21))
    for i in range(len(seq) - 2):
        vectors[i][coden_dict1[seq[i:i + 3].replace('T', 'T')]] = 1
    return vectors.tolist()#矩阵转换为列表


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('T', 'T')
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array


# one-hot编码
def bpf(seq):
    phys_dic = {
        'A': [0, 0, 0, 1],
        'T': [0, 0, 1, 0],
        'C': [0, 1, 0, 0],
        'G': [1, 0, 0, 0]}
    seqLength = len(seq)
    # seqLength = 3000
    sequence_vector = np.zeros([5000, 4])
    for i in range(0, seqLength):
        sequence_vector[i, 0:4] = phys_dic[seq[i]]
    for i in range(seqLength, 5000):
        sequence_vector[i, -1] = 1
    return sequence_vector


# NCP编码
def processFastaFile(seq):
    phys_dic = {
        'A': [1, 1, 1],
        'T': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]}
    seqLength = len(seq)
    # seqLength = 3000
    sequence_vector = np.zeros([5000, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 5000):
        sequence_vector[i, -1] = 1
    return sequence_vector


# TPCP编码
def tpcp(seq):
    phys_dic = {
        # Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AT': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14, ],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
        'TA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'TT': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'TC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'TG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
        'CA': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CT': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GT': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34]}

    seqLength = len(seq)
    sequence_vector = np.zeros([5000, 11])
    k = 2
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i + k]]
    return sequence_vector


def nd(seq, seq_length):
    seq = seq.strip()
    nd_list = [None] * seq_length
    for j in range(seq_length):
        # print(seq[0:j]) 第j个碱基在前j+1个碱基中的占比
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j + 1].count('A') / (j + 1), 3)
        elif seq[j] == 'T':
            nd_list[j] = round(seq[0:j + 1].count('T') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j + 1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j + 1].count('G') / (j + 1), 3)
    return np.array(nd_list)


def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    for seq in seq_dict.values():
        seq = seq.replace('T', 'T')
        bag_sen.append(seq)

    return np.asarray(bag_sen)


def dealwithdata(cell_line, genomics):
    seq_length = 5000
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    tris4 = get_4_trids()
    dataX = []
    dataY = []
    with open('data/' + cell_line + '/positive') as f:
        for line in f:
            if '>' not in line:
                line = line.strip()
                # print('每行序列长度 :', len(line.strip()))
                probMatr = processFastaFile(line.strip())   # shape(5000, 3)
                probMatr_ND = nd(line.strip(), seq_length)  # shape(5000)
                probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))    # shape(5000,4)将计算的ND拼接在Matr后面
                kmer1 = coden(line.strip(), 1, tris1)  # shape(5000, 4)
                kmer2 = coden(line.strip(), 2, tris2)  # shape(5000,16)
                kmer3 = coden1(line.strip())    # shape(5000, 21)
                Kmer = np.hstack((kmer1, kmer2, kmer3))  # shape(5000, 41)
                Feature_Encoding = np.column_stack((probMatr_NDCP, Kmer))  # shape(5000, 4+41=45)
                dataX.append(Feature_Encoding.tolist())  # 一行一行序列处理, 最终shape(2896,5000,45)
                dataY.append([0,1])
    print("positive end!")
    with open('data/' + cell_line + '/negative') as f:
        for line in f:
            if '>' not in line:
                line = line.strip()
                probMatr = processFastaFile(line.strip())
                probMatr_ND = nd(line.strip(), seq_length)
                probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))
                kmer1 = coden(line.strip(), 1, tris1)  # shape(5000, 4)
                kmer2 = coden(line.strip(), 2, tris2)  # shape(5000,16)
                kmer3 = coden1(line.strip())    # shape(5000, 21)
                Kmer = np.hstack((kmer1, kmer2, kmer3))  # shape(5000, 41)
                Feature_Encoding = np.column_stack((probMatr_NDCP, Kmer))  # shape(5000, 4+41=45)
                dataX.append(Feature_Encoding.tolist())  # shape(2896,5000,45)
                dataY.append([1,0])
    print("negative end!")
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)   # 为了打乱顺序 choice(5792, 5792, replace=False)
    dataX = np.array(dataX)[indexes]    # shape(5792, 5000, 45)
    dataY = np.array(dataY)[indexes]
    genomics = genomics[indexes]
    print('dataX.shape:', dataX.shape)
    print('dataY.shape:', dataY.shape)
    print('genomics.shape:', genomics.shape)
    # np.save('data/' + cell_line + '/dataX.npy', dataX)
    # np.save('data/' + cell_line + '/dataY.npy', dataY)
    # np.save('data/' + cell_line + '/genomics.npy', genomics)
    train_val_X, test_X, train_val_y, test_y, train_val_gen, test_gen = train_test_split(dataX, dataY, genomics,
                                                                                         test_size=0.2)
    return train_val_X, test_X, train_val_y, test_y, train_val_gen, test_gen
