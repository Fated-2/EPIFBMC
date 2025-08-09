import tensorflow as tf
from tensorflow.keras import layers
import math

class MANet(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MANet, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # assert d_model % num_heads == 0
        # self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size, time, dimension = tf.unstack(tf.shape(q))
        n_d = self.d_model // self.num_heads
        q, k, v = self.wq(q), self.wk(k), self.wv(v)  # (batch_size, seq_len, d_model)
        q = tf.transpose(tf.reshape(q, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        k_t = tf.transpose(k, perm=[0, 1, 3, 2])
        score = tf.matmul(q, k_t) / math.sqrt(n_d)
        mask = tf.linalg.band_part(tf.ones([time, time], dtype=tf.bool), -1, 0)  # 生成一个下三角矩阵，其余值为 0
        score = tf.where(mask, score, float('-inf'))
        score = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(score, v)

        output = tf.reshape(tf.transpose(output, perm=[0, 2, 1, 3]), (batch_size, time, n_d*self.num_heads))
        output = self.dense(output)  # (batch_size, seq_len_q, d_model)

        return output


class genomics_attention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(genomics_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # assert d_model % num_heads == 0
        # self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size, time, dimension = tf.unstack(tf.shape(q))
        n_d = self.d_model // self.num_heads
        q, k, v = self.wq(q), self.wk(k), self.wv(v)  # (batch_size, seq_len, d_model)
        q = tf.transpose(tf.reshape(q, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (batch_size, time, self.num_heads, n_d)), perm=[0, 2, 1, 3])
        # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        k_t = tf.transpose(k, perm=[0, 1, 3, 2])
        score = tf.matmul(q, k_t) / math.sqrt(n_d)
        mask = tf.linalg.band_part(tf.ones([time, time], dtype=tf.bool), -1, 0)  # 生成一个下三角矩阵，其余值为 0
        score = tf.where(mask, score, float('-inf'))
        score = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(score, v)

        output = tf.reshape(tf.transpose(output, perm=[0, 2, 1, 3]), (batch_size, time, n_d*self.num_heads))
        output = self.dense(output)  # (batch_size, seq_len_q, d_model)

        return output


def MCNet(input_shape_sequence, input_shape_genomics):
    sequence_input = layers.Input(shape=input_shape_sequence)
    genomics_input = layers.Input(shape=input_shape_genomics)

    x = layers.Conv1D(64, kernel_size=9, strides=1, activation='relu')(sequence_input)  # shape(,5000-9+1=4992,32)
    x = layers.AveragePooling1D(pool_size=3, strides=3)(x)  # shape(,4992/3=1664,32)
    # x = layers.Conv1D(128, kernel_size=5, strides=1, activation='relu')(x)  # shape(,1664-5+1=1660,64)
    # x = layers.AveragePooling1D(pool_size=2, strides=2)(x)  # shape(,1660/3=553,64)
    x = layers.Dropout(0.5)(x)  # shape(,553,64)

    x1 = layers.Conv1D(32, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x2 = layers.Conv1D(32, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x3 = layers.Conv1D(32, kernel_size=5, strides=1, padding='same', use_bias=False)(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Dropout(0.2)(x3)
    x3 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    x = tf.concat([x1, x2, x3], axis=2)
    # x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu')(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    # output = layers.MultiHeadAttention(8, 128)(x, x)
    output = MANet(128, 8)(x, x, x)
    output = layers.GlobalAveragePooling1D()(output)
    # 对输入的基因组特征数据进行批量归一化, 加速训练过程, 提高泛化能力
    merge2 = layers.BatchNormalization()(genomics_input)
    merge2 = layers.Dropout(0.5)(merge2)
    merge2 = layers.Dense(128, activation='relu')(merge2)
    # 合并正反向DNA序列和基因组特征
    merge2 = layers.Concatenate(axis=1)([output, merge2])

    output = layers.Dense(2, activation='softmax')(merge2)

    model = tf.keras.models.Model(inputs=[sequence_input, genomics_input], outputs=output)
    # print('打印模型主要信息 : ', model.summary())  # 打印模型的主要信息, 层的数量, 参数数量
    return model

