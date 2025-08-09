import matplotlib
matplotlib.use('Agg')  # 不尝试打开一个窗口来显示图形,在脚本顶部添加这行
import numpy as np
import keras
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt

import data_load
import Four-encoding
from model import *

import time
import gc
start_time = time.perf_counter()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
device = tf.config.experimental.list_physical_devices("GPU")


class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        # 获取模型的初始参数值
        weights = self.model.get_weights()
        for i, w in enumerate(weights):
            print("Layer {}: \n{}".format(i, w))

    def on_batch_end(self, batch, logs=None):
        # 获取每个 batch 结束时参数的更新情况
        weights = self.model.get_weights()
        for i, w in enumerate(weights):
            print("Layer {}: \n{}".format(i, w))


cell_line = 'IMR90'  # 要处理的细胞系

# x_sequence增强子序列 y_sequence启动子序列
x_sequence, y_sequence = data_load.load(cell_line)
x_sequence = np.array(x_sequence)
y_sequence = np.array(y_sequence)
genomics = data_load.save_to_file(x_sequence, y_sequence, cell_line)
train_val_X, test_X, train_val_y, test_y, train_val_gen, test_gen = encoding.dealwithdata(cell_line, genomics)

test_y = test_y[:, 1]
print('测试集长度 :', len(test_y))
input_shape_sequence = (5000, 45)
input_shape_genomics = (1148,)

# del dataX, dataY, genomics, train_val_X, train_val_gen
gc.collect()

k = 5
i = 0
# 测试集评估指标
# 创建一个空列表，用于存储每个折叠的FPR、TPR值和AUC值。
fpr_list = []
tpr_list = []
aucs = []  # 计算ROC曲线下的面积
Acc = []  # 准确率
Mcc = []
precision1 = []
recall1 = []  # 计算召回率 : 比值“tp / (tp + fn)”
fscore1 = []


def plot_fun(history):
    # 绘制训练集和验证集的 loss 曲线
    plt.plot(history.history['loss'], label='The' + str(len(fpr_list)) + 'th train_loss')
    plt.plot(history.history['val_loss'], label='The' + str(len(fpr_list)) + 'th val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./pictures/' + cell_line + '/' + str(len(fpr_list)) + 'fold' + '.png')

    # 训练完成后，使用测试集进行预测，并计算预测结果的AUC值。
    predictions = model.predict([test_X, test_gen])[:, 1]
    print('The', len(fpr_list), 'th predictions : ', predictions)
    pre = np.argmax(model.predict([test_X, test_gen]), axis=-1)

    fpr, tpr, _ = roc_curve(test_y, predictions)
    auc = roc_auc_score(test_y, predictions)  # 或使用auc = auc(fpr, tpr)也可以计算准确率
    precision, recall, thresholds = precision_recall_curve(test_y, predictions)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AP = average_precision_score(test_y, predictions)

    precision = precision_score(test_y, pre)
    recall = recall_score(test_y, pre)
    fscore = f1_score(test_y, pre)
    acc = accuracy_score(test_y, pre)
    mcc = matthews_corrcoef(test_y, pre)

    aucs.append(auc)
    Acc.append(acc)
    Mcc.append(mcc)
    precision1.append(precision)
    recall1.append(recall)
    fscore1.append(fscore)
    # 保存ROC数据
    if len(fpr_list) == k:
        roc_data = np.column_stack((fpr_list[-1], tpr_list[-1]))
        np.savetxt('./result/{}(ACC={:.4f}).txt'.format(cell_line, np.mean(Acc)), roc_data, delimiter='\t')

    print('aucs:', np.mean(aucs), aucs, cell_line)  # 输出平均AUC值和蛋白质名称。
    print('acc:', np.mean(Acc), Acc, cell_line)
    print('mcc:', np.mean(Mcc), Mcc, cell_line)
    print('precision:', np.mean(precision1), precision1, cell_line)
    print('recall:', np.mean(recall1), recall1, cell_line)
    print('f1-score:', np.mean(fscore1), fscore1, cell_line)

# 训练参数
MAX_EPOCH = 100  # 模型训练的最大轮数, 若模型在验证集(测试集)上的性能不再提升，训练可能会提前终止。
BATCH_SIZE = 64  # 每次训练迭代中用于更新模型权重的样本数量 , 只决定训练的速度, 而不影响验证集的性能, 理论上所有的batch_size能获得一样的效果
learning_rate = 1e-4  # 学习率为0.0001

gkf = KFold(n_splits=k)  # n_splits指定了交叉验证的折数(代表数据被分成几部分)
# split方法返回一个迭代器, 每次迭代产生一个训练集和测试集的索引划分
# 这个for循环的次数取决于n_splits的值, 这里是10次
for index_train, index_val in gkf.split(train_val_y):
    print('训练集长度 : ', len(index_train))  # index_train是个多个索引的数组, 用于训练的数据的下标
    print('验证集长度 : ', len(index_val))
    #

    print('configure cnn network')

    model = MCANet(input_shape_sequence, input_shape_genomics)
    
    # 训练过程的编译模型, 定义了该模型在训练过程中的损失函数、优化器、评估模型(准确率为指标)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    print('model summary')
    model.summary()
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
    # 使用自定义回调函数
    custom_callback = CustomCallback()

    print('model training')
    history = model.fit(x=[train_val_X[index_train], train_val_gen[index_train] ], y=train_val_y[index_train], batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCH, validation_data=([train_val_X[index_val], train_val_gen[index_val]], train_val_y[index_val] ),
                        callbacks=[early_stopping_monitor])
    plot_fun(history)  # 存入函数体

    '''
    # 绘制训练集和验证集的 loss 曲线
    plt.plot(history.history['loss'], label='The' + str(len(fpr_list)) + 'th train_loss')
    plt.plot(history.history['val_loss'], label='The' + str(len(fpr_list)) + 'th val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./pictures/' + str(len(fpr_list)) + 'fold-' + cell_line + '.png')

    # 训练完成后，使用测试集进行预测，并计算预测结果的AUC值。
    predictions = model.predict([test_X, test_gen])[:, 1]
    print('The', len(fpr_list), 'th predictions : ', predictions)
    pre = np.argmax(model.predict([test_X, test_gen]), axis=-1)

    fpr, tpr, _ = roc_curve(test_y, predictions)
    auc = roc_auc_score(test_y, predictions)  # 或使用auc = auc(fpr, tpr)也可以计算准确率
    precision, recall, thresholds = precision_recall_curve(test_y, predictions)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AP = average_precision_score(test_y, predictions)

    precision = precision_score(test_y, pre)
    recall = recall_score(test_y, pre)
    fscore = f1_score(test_y, pre)
    acc = accuracy_score(test_y, pre)

    aucs.append(auc)
    Acc.append(acc)
    precision1.append(precision)
    recall1.append(recall)
    fscore1.append(fscore)
    # 保存ROC数据
    if len(fpr_list) == k:
        roc_data = np.column_stack((fpr_list[-1], tpr_list[-1]))
        np.savetxt('./result/{}(ACC={:.4f}).txt'.format(cell_line, np.mean(Acc)), roc_data, delimiter='\t')

    print('aucs:', np.mean(aucs), aucs, cell_line)  # 输出平均AUC值和蛋白质名称。
    print('acc:', np.mean(Acc), Acc, cell_line)
    print('precision:', np.mean(precision1), precision1, cell_line)
    print('recall:', np.mean(recall1), recall1, cell_line)
    print('f1-score:', np.mean(fscore1), fscore1, cell_line)
    '''

end_time = time.perf_counter()  # 程序结束时间
run_time = end_time - start_time
print(f"程序运行时间: {run_time} 秒")
