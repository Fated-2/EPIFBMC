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
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
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
    input_data_x_forward = Input(shape=input_shape_x_forward)
    input_data_x_reverse = Input(shape=input_shape_x_reverse)
    input_data_y_forward = Input(shape=input_shape_y_forward)
    input_data_y_reverse = Input(shape=input_shape_y_reverse)
    input_data_genomics = Input(shape=input_shape_genomics)

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
    merge1 = Dropout(0.5)(merge1)
    merge1 = Dense(32, activation='relu')(merge1)

    merge2 = BatchNormalization()(input_data_genomics)
    merge2 = Dropout(0.5)(merge2)
    merge2 = Dense(128, activation='relu')(merge2)
    merge2 = Concatenate(axis=1)([merge1, merge2])

    # output = Dense(16, activation='sigmoid')(merge2)
    output = Dense(1, activation='sigmoid')(merge2)

    model = Model(
        [input_data_x_forward, input_data_x_reverse, input_data_y_forward, input_data_y_reverse, input_data_genomics],
        output)
    print(model.summary())
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
