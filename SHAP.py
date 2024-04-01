import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
import sys
import shap
import data_load
import feature_code
import model
from sklearn.linear_model import LogisticRegression
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.backend import get_session
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

cell_line = 'GM12878'
genomics = np.loadtxt('feature/' + cell_line + '/genomics.csv', delimiter=',')
input_shape_genomics = (len(genomics[0]),)


def get_name():
    name = []
    f = open('data/' + cell_line + '/x.bed')
    for i in f.readlines():
        if i[0] != ' ':
            name.append(i.strip().split('\t')[0])
    f.close()
    return name


chr_name = get_name()
k = 10
i = 0
MAX_EPOCH = 50
BATCH_SIZE = 64
learning_rate = 3e-4
label = np.loadtxt('data/' + cell_line + '/label.txt')
gkf = GroupKFold(n_splits=k)

global shap_score_all, label_all

for index_train, index_test in gkf.split(label, groups=chr_name):
    print(len(index_train))
    print(len(index_test))

    # 训练集
    genomics_train = genomics[index_train]
    label_train = label[index_train]

    # 分成正负两类
    num_pos = np.sum(label_train == 1)
    num_neg = np.sum(label_train == 0)
    print(num_pos, num_neg)
    ratio = int(num_neg / num_pos)
    index_pos = np.where(label_train == 1)
    index_neg = np.where(label_train == 0)
    print('ratio:', ratio)

    # 正的数据
    genomics_train_pos = genomics_train[index_pos]
    label_train_pos = label_train[index_pos]
    # 负的数据
    genomics_train_neg_total = genomics_train[index_neg]
    label_train_neg_total = label_train[index_neg]
    # 测试集
    genomics_test = genomics[index_test]
    label_test = label[index_test]
    # 平衡测试集
    num = np.arange(0, len(label_test)).reshape(-1, 1)
    # print(num.shape)
    ros = RandomUnderSampler(random_state=0)
    num, label_test_bal = ros.fit_resample(num, label_test)
    num = np.squeeze(num).tolist()
    genomics_test_bal = genomics_test[num]

    model_score = np.zeros((ratio, len(label_test)))
    model_score_bal = np.zeros((ratio, len(label_test_bal)))
    j = 0
    kf = KFold(n_splits=ratio, shuffle=True, random_state=0)
    # 声明SHAP值变量
    number_fold = len(label_test_bal)
    shap_score_fold_ratio = np.zeros((ratio, number_fold, len(genomics[0])))
    label_fold = label_test_bal

    for _, index in kf.split(label_train_neg_total):
        # print(index)
        genomics_train_neg = genomics_train_neg_total[index]
        label_train_neg = label_train_neg_total[index]

        genomics_train_kf = np.concatenate((genomics_train_pos, genomics_train_neg), axis=0)
        label_train_kf = np.concatenate((label_train_pos, label_train_neg))
        # 分割训练集和验证集
        genomics_train_kf, genomics_val_kf, label_train_kf, label_val_kf = train_test_split(genomics_train_kf,
                                                                                            label_train_kf,
                                                                                            test_size=0.1,
                                                                                            random_state=0)

        early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
        cnn = model.IChrom_genomics(input_shape_genomics)
        cnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        cnn.fit(x=genomics_train_kf, y=label_train_kf, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, validation_data=(
            genomics_val_kf, label_val_kf), callbacks=[early_stopping_monitor])
        model_score[j] = np.squeeze(cnn.predict(genomics_test))
        model_score_bal[j] = np.squeeze(cnn.predict(genomics_test_bal))

        shap.initjs()
        # background = x_train[np.random.choice(x_train.shape[0], 2000, replace=False)]
        explainer = shap.DeepExplainer(cnn, genomics_train_kf)
        shap_values = explainer.shap_values(genomics_test_bal)
        # shap.summary_plot(shap_values, genomics_test_bal, max_display=10) # 特征重要性
        shap_score_fold_ratio[j] = np.squeeze(np.array(shap_values))
        # print(np.array(shap_values).shape)
        # shap.summary_plot(shap_score_fold_ratio[j], genomics_test_bal, max_display=10)

        j = j + 1

    shap_score_fold = np.mean(shap_score_fold_ratio, axis=0)

    np.savetxt('SHAP/' + cell_line + '/shap_score_' + str(i) + '.txt', shap_score_fold)
    np.savetxt('SHAP/' + cell_line + '/label_' + str(i) + '.txt', label_fold)
    if i == 0:
        shap_score_all = shap_score_fold
        label_all = label_fold
    else:
        shap_score_all = np.concatenate((shap_score_all, shap_score_fold), axis=0)
        label_all = np.concatenate((label_all, label_fold))

    i = i + 1

print(shap_score_all.shape)
print(label_all.shape)
# np.savetxt('SHAP/' + cell_line + '/shap_score_all.txt', shap_score_all)
# np.savetxt('SHAP/' + cell_line + '/label_all.txt', label_all)
