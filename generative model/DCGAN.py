from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Reshape, LeakyReLU, Conv1D, UpSampling1D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import heapq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def nor_train_test(x_train, x_test):
    x = np.concatenate((x_train, x_test), axis=0)
    x = noramlization(x)
    x_train = x[0:len(x_train)]
    x_test = x[len(x_train):]
    return x_train, x_test


def generator_model():
    model = Sequential()
    model.add(Dense(data_dim, input_shape=(character,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Reshape((data_dim, 1)))
    # model.add(UpSampling1D())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(character, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=1, kernel_size=3, padding='same', activation='tanh'))

    '''model.add(Dense(64, input_shape=(character,)))
    model.add(LeakyReLU(0.2))
    model.add(Dense(data_dim, input_shape=(character,)))
    model.add(Activation('tanh'))'''

    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(data_dim, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    '''model.add(Dense(64, input_shape=(data_dim,)))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, input_shape=(data_dim,)))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))'''

    return model


def combine(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    return model


# frist, pls load data

data_dim = x_train.shape[1]
character = 100

g = generator_model()
g.summary()
d = discriminator_model()
d.summary()
g_d = combine(g, d)
g_d.summary()

result_path = 'generated_data/result/'
model_path = 'Model/'
generated_data_path = 'generated_data/'

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(generated_data_path):
    os.makedirs(generated_data_path)


def generated(noise_need, name):
    g = generator_model()
    try:
        g.load_weights(model_path + "generatorA")
        print("load weight")
    except:
        print("no weight")
    noise_need = np.random.normal(size=(1, character))
    generated_data_need = g.predict(noise_need, verbose=0)
    # data = combine_datas(generated_data_need)
    data = generated_data_need
    data = data * 0.5 + 0.5
    data = np.squeeze(data)
    np.savetxt(result_path + name + '.txt', data)


def train(BATCH_SIZE, X_train):
    global best
    generated_data_size = number * 9
    # print(X_train.max())
    X_train = ((X_train.astype(np.float32)) - 0.5) / 0.5  # -1~1
    # print(X_train.max())
    # 模型及其优化器
    d = discriminator_model()
    g = generator_model()
    g_d = combine(g, d)
    '''d_optimizer = RMSprop(learning_rate=0.0003)
    g_optimizer = SGD(learning_rate=0.001)'''
    d_optimizer = Adam(learning_rate=0.0002)
    g_optimizer = SGD(learning_rate=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))  # 生成器
    g_d.compile(loss='binary_crossentropy', optimizer=g_optimizer)  # 联合模型
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)  # 判别器

    # 导入权重
    try:
        d.load_weights(model_path + "discriminatorA")
        print("判别器权重导入成功")
        g.load_weights(model_path + "generatorA")
        print("生成器权重导入成功")
    except:
        print("无权重")

    for epoch in range(1000):
        # 每1轮打印一次当前轮数
        if epoch % 1 == 0:
            print('Epoch is ', epoch + 1)
        for index in range(X_train.shape[0] // BATCH_SIZE):
            # 产生（-1，1）的正态分布的维度为（BATCH_SIZE, character）的矩阵
            noise = np.random.normal(0, 1 / 3, size=(BATCH_SIZE, character))
            train_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_data = g.predict(noise, verbose=0)

            if index % 20 == 0:
                # 每20次输出一次数据
                noise_need = np.random.normal(0, 1 / 3, size=(generated_data_size, character))
                generated_data_need = g.predict(noise_need, verbose=0)
                # data = combine_datas(generated_data_need)
                data = generated_data_need
                data = data * 0.5 + 0.5  # 0~1
                data = np.squeeze(data)
                acc_average = 0
                for i in range(9):
                    model = KNeighborsClassifier(n_neighbors=1)
                    x1 = np.squeeze((X_train * 0.5 + 0.5))
                    x2 = data[i * number:(i + 1) * number]
                    y1 = np.ones(number)
                    y2 = np.zeros(number)
                    x = np.concatenate((x1, x2), axis=0)
                    y = np.concatenate((y1, y2), axis=0)
                    t = np.expand_dims(x2, 2)
                    kf = KFold(10, True, 10)
                    i = 0
                    acc_model = np.zeros(10)
                    for train_index, test_index in kf.split(x):
                        model.fit(x[train_index], y[train_index])
                        model_pred = model.predict_proba(x[test_index])
                        model_predict = model_pred[:, 1]
                        model_p = model.predict(x[test_index])
                        acc_model[i] = accuracy_score(y[test_index], model_p)
                        i = i + 1
                    acc_average += acc_model.mean()
                print('average knn result:', acc_average / 9)
                if acc_average / 9 < best:

                    best = acc_average / 9
                    g.save_weights(model_path + 'best/generatorA', True)
                    d.save_weights(model_path + 'best/discriminatorA', True)
                    np.savetxt(model_path + 'best/generate_best_data.txt', data)
                # np.savetxt(generated_data_path + 'generate_data.txt', data)

            # 每运行一次训练一次判别器
            if epoch >= 2:
                if index % 1 == 0:
                    X = np.concatenate((train_batch, generated_data))
                    Y = list((np.random.rand(BATCH_SIZE) * 10 + 90) / 100) + [0] * BATCH_SIZE
                    d_loss = d.train_on_batch(X, Y)

            noise = np.random.normal(size=(BATCH_SIZE, character))
            d.trainable = False
            g_loss = g_d.train_on_batch(noise, list((np.random.rand(BATCH_SIZE) * 10 + 90) / 100))
            d.trainable = True
            if epoch > 5:
                if index % 10 == 0:
                    print('batch: %d, g_loss: %f, d_loss: %f' % (index, g_loss, d_loss))

            if index % 10 == 0:
                g.save_weights(model_path + 'generatorA', True)
                print('Successfully save generatorA')
                d.save_weights(model_path + 'discriminatorA', True)
                print('Successfully save discriminatorA')


best = 1.0
train(BATCH_SIZE=100, X_train=x_train)
print(best)
