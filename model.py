from keras import Input, Model
from keras.layers import Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Dropout, Layer, Reshape, Concatenate, \
    LeakyReLU, GlobalAveragePooling1D, GlobalMaxPooling1D, ReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
from keras.layers import GlobalMaxPooling1D
from keras import backend as K
from keras import initializers
from keras.layers import LSTM, BatchNormalization


class AttLayer(Layer):
    """
    自定义的注意力机制层, 继承了Keras的Layer类
    通过学习输入序列中每个元素的重要性来加权输入, 然后对加权后的输入进行求和,得到最终输出
    """

    # 初始化(构造函数), 接受了一个参数attention_dim指定注意力层的维度(用于表示注意力权重的向量的维度大小)
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True  # 支持掩码, 掩码可用来指示序列中哪些位置是有效的
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()  # 调用执行父类的初始化方法

    # 初始化层的权重, 这些权重在模型训练过程中会被学习和更新
    def build(self, input_shape):
        assert len(input_shape) == 3  # 输入形状的长度是3, 代表批次大小、时间不长和特征数量
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))  # 创建一个权重矩阵W
        self.b = K.variable(self.init((self.attention_dim,)))  # 创建一个偏置向量b, 形状即注意力维度
        self.u = K.variable(self.init((self.attention_dim, 1)))  # 创建一个权重矩阵, 形状(注意力维度, 1), 用于计算注意力分数
        self.trainable_weights = [self.W, self.b, self.u]  # 这个列表会在训练过程中被优化器用来更新权重
        super(AttLayer, self).build(input_shape)  # 调用父类的build, 正确完成层的构建过程

    # 当输入数据具有掩码时, 才会被调用, mask是可选参数, 表示输入数据的掩码
    def compute_mask(self, inputs, mask=None):
        return mask

    # 通过计算输入数据x的加权和来生成输出, x的形状为 [batch_size, sel_len, attention_dim]，
    # 其中 batch_size 是批次大小，sel_len 是序列长度，attention_dim 是注意力维度
    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        # 先计算输入x与权重矩阵W的点积, 再加上偏置b, 再用Tanh 激活函数来获得中间表示uit
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        # 计算注意力分数, ait代表注意力权重
        ait = K.dot(uit, self.u)  # 将中间表示与权重向量点积, 得到注意力分数的原始值
        ait = K.squeeze(ait, -1)
        # 应用掩码
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        # 归一化注意力权重
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # 加权输入
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        # 计算输出, 通过这一函数模型就可以学习输入序列中不同部分的重要性, 并在生成输出时给予不同的权重
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# focal loss
def binary_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def IChrom_seq(input_shape_x_forward, input_shape_x_reverse, input_shape_y_forward, input_shape_y_reverse):
    input_data_x_forward = Input(shape=input_shape_x_forward)
    input_data_x_reverse = Input(shape=input_shape_x_reverse)
    input_data_y_forward = Input(shape=input_shape_y_forward)
    input_data_y_reverse = Input(shape=input_shape_y_reverse)

    x_forward = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_x_forward)
    x_forward = MaxPooling1D(pool_size=3, strides=3)(x_forward)
    x_forward = Conv1D(64, kernel_size=5, strides=1, activation='relu')(x_forward)
    x_forward = MaxPooling1D(pool_size=3, strides=3)(x_forward)
    x_forward = Dropout(0.2)(x_forward)

    x_reverse = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_x_reverse)
    x_reverse = MaxPooling1D(pool_size=3, strides=3)(x_reverse)
    x_reverse = Conv1D(64, kernel_size=5, strides=1, activation='relu')(x_reverse)
    x_reverse = MaxPooling1D(pool_size=3, strides=3)(x_reverse)
    x_reverse = Dropout(0.2)(x_reverse)

    y_forward = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_y_forward)
    y_forward = MaxPooling1D(pool_size=3, strides=3)(y_forward)
    y_forward = Conv1D(64, kernel_size=5, strides=1, activation='relu')(y_forward)
    y_forward = MaxPooling1D(pool_size=3, strides=3)(y_forward)
    y_forward = Dropout(0.2)(y_forward)

    y_reverse = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_y_reverse)
    y_reverse = MaxPooling1D(pool_size=3, strides=3)(y_reverse)
    y_reverse = Conv1D(64, kernel_size=5, strides=1, activation='relu')(y_reverse)
    y_reverse = MaxPooling1D(pool_size=3, strides=3)(y_reverse)
    y_reverse = Dropout(0.2)(y_reverse)

    merge1 = Concatenate(axis=1)([x_forward, x_reverse, y_forward, y_reverse])
    merge1 = AttLayer(50)(merge1)
    # merge1 = Flatten()(merge1)
    merge1 = Dropout(0.5)(merge1)
    merge1 = Dense(32, activation='relu')(merge1)
    output = Dense(1, activation='sigmoid')(merge1)
    model = Model([input_data_x_forward, input_data_x_reverse, input_data_y_forward, input_data_y_reverse], output)
    print(model.summary())
    return model


def IChrom_deep(input_shape_x_forward, input_shape_x_reverse, input_shape_y_forward, input_shape_y_reverse,
                input_shape_genomics):
    """
    创建DL模型即定义模型结构/架构, 用于处理染色质相互作用的预测任务. 使用了CNN和注意力机制, 并结合基因组特征

    :param input_shape_x_forward: 5个输入层, 正向DNA序列
    :param input_shape_x_reverse: 反向DNA序列
    :param input_shape_y_forward:
    :param input_shape_y_reverse:
    :param input_shape_genomics: 基因组特征
    :return: 返回建好的模型
    """
    input_data_x_forward = Input(shape=input_shape_x_forward)
    input_data_x_reverse = Input(shape=input_shape_x_reverse)
    input_data_y_forward = Input(shape=input_shape_y_forward)
    input_data_y_reverse = Input(shape=input_shape_y_reverse)
    input_data_genomics = Input(shape=input_shape_genomics)

    # 使用2个卷积层(Conv1D有32个过滤器, Conv1D有64个过滤器), 激活函数为relu
    # 使用MaxPooling1D最大池化层, 降低维度, 保留重要特征
    # Dropout层, 防止过拟合, 丢弃一部分神经元的输出
    x_forward = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_x_forward)
    x_forward = MaxPooling1D(pool_size=3, strides=3)(x_forward)
    x_forward = Conv1D(64, kernel_size=5, strides=1, activation='relu')(x_forward)
    x_forward = MaxPooling1D(pool_size=3, strides=3)(x_forward)
    x_forward = Dropout(0.2)(x_forward)

    x_reverse = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_x_reverse)
    x_reverse = MaxPooling1D(pool_size=3, strides=3)(x_reverse)
    x_reverse = Conv1D(64, kernel_size=5, strides=1, activation='relu')(x_reverse)
    x_reverse = MaxPooling1D(pool_size=3, strides=3)(x_reverse)
    x_reverse = Dropout(0.2)(x_reverse)

    y_forward = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_y_forward)
    y_forward = MaxPooling1D(pool_size=3, strides=3)(y_forward)
    y_forward = Conv1D(64, kernel_size=5, strides=1, activation='relu')(y_forward)
    y_forward = MaxPooling1D(pool_size=3, strides=3)(y_forward)
    y_forward = Dropout(0.2)(y_forward)

    y_reverse = Conv1D(32, kernel_size=9, strides=1, activation='relu')(input_data_y_reverse)
    y_reverse = MaxPooling1D(pool_size=3, strides=3)(y_reverse)
    y_reverse = Conv1D(64, kernel_size=5, strides=1, activation='relu')(y_reverse)  # ①卷积层
    y_reverse = MaxPooling1D(pool_size=3, strides=3)(y_reverse)  # ②最大池化层
    y_reverse = Dropout(0.2)(y_reverse)  # ③正则化技术, 丢弃一些神经元的输出

    # ④拼接层, 将不同卷积层的特征合并在一起
    merge1 = Concatenate(axis=1)([x_forward, x_reverse, y_forward, y_reverse])
    merge1 = AttLayer(50)(merge1)  # 自定义注意力机制
    merge1 = Dropout(0.5)(merge1)
    merge1 = Dense(32, activation='relu')(merge1)  # ⑤全连接层, 用于进一步处理合并后的特征, 指定神经元的数量

    # 对输入的基因组特征数据进行批量归一化, 加速训练过程, 提高泛化能力
    merge2 = BatchNormalization()(input_data_genomics)
    merge2 = Dropout(0.5)(merge2)
    merge2 = Dense(128, activation='relu')(merge2)
    # 合并正反向DNA序列和基因组特征
    merge2 = Concatenate(axis=1)([merge1, merge2])

    # output = Dense(16, activation='sigmoid')(merge2)
    # 输出层, 使用sigmoid激活函数, 输出值介于0-1, 表示染色质相互作用的概率
    output = Dense(1, activation='sigmoid')(merge2)

    # 使用Keras 的 Model 类来编译模型
    model = Model(
        [input_data_x_forward, input_data_x_reverse, input_data_y_forward, input_data_y_reverse, input_data_genomics],
        output)
    print('打印模型主要信息 : ', model.summary())  # 打印模型的主要信息, 层的数量, 参数数量
    return model


def IChrom_genomics(input_shape_genomics):
    input_data_genomics = Input(shape=input_shape_genomics)

    merge2 = BatchNormalization()(input_data_genomics)
    merge2 = Dropout(0.5)(merge2)
    merge2 = Dense(128, activation='relu')(merge2)

    # output = Dense(16, activation='sigmoid')(merge2)
    output = Dense(1, activation='sigmoid')(merge2)

    model = Model(input_data_genomics, output)
    print(model.summary())
    return model
