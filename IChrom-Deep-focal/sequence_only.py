import numpy as np
import data_load
import feature_code
import model
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model

cell_line = 'GM12878'
x_forward, x_reverse, y_forward, y_reverse = data_load.load_Bi(cell_line)
x_forward_feature = feature_code.one_hot(x_forward)
x_reverse_feature = feature_code.one_hot(x_reverse)
y_forward_feature = feature_code.one_hot(y_forward)
y_reverse_feature = feature_code.one_hot(y_reverse)
input_shape_x_forward = (5000, 4)
input_shape_x_reverse = (5000, 4)
input_shape_y_forward = (5000, 4)
input_shape_y_reverse = (5000, 4)


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

# 测试集指标
auprc_bal = np.zeros(k)
acc_bal = np.zeros(k)
mcc_bal = np.zeros(k)
precision_bal = np.zeros(k)
recall_bal = np.zeros(k)
f1_bal = np.zeros(k)

MAX_EPOCH = 50
BATCH_SIZE = 64
learning_rate = 3e-4

label = np.loadtxt('data/' + cell_line + '/label.txt')
gkf = GroupKFold(n_splits=k)
kf = KFold(10, True)
# for index_train, index_test in kf.split(label):
for index_train, index_test in gkf.split(label, groups=chr_name):
    print(len(index_train))
    print(len(index_test))

    # 训练集
    x_forward_feature_train = x_forward_feature[index_train, :, :]
    x_reverse_feature_train = x_reverse_feature[index_train, :, :]
    y_forward_feature_train = y_forward_feature[index_train, :, :]
    y_reverse_feature_train = y_reverse_feature[index_train, :, :]
    label_train = label[index_train]

    # 分割训练集和验证集
    x_forward_feature_train, x_forward_feature_val, x_reverse_feature_train, x_reverse_feature_val, \
    y_forward_feature_train, y_forward_feature_val, y_reverse_feature_train, y_reverse_feature_val, \
    label_train, label_val = train_test_split(
        x_forward_feature_train, x_reverse_feature_train, y_forward_feature_train, y_reverse_feature_train, label_train,
        test_size=0.1, random_state=0)

    # 测试集
    x_forward_feature_test = x_forward_feature[index_test, :, :]
    x_reverse_feature_test = x_reverse_feature[index_test, :, :]
    y_forward_feature_test = y_forward_feature[index_test, :, :]
    y_reverse_feature_test = y_reverse_feature[index_test, :, :]
    label_test = label[index_test]

    # 平衡测试集
    num = np.arange(0, len(label_test)).reshape(-1, 1)
    print(num.shape)
    ros = RandomUnderSampler()
    num, label_test_bal = ros.fit_resample(num, label_test)
    num = np.squeeze(num).tolist()
    x_forward_feature_test_bal = x_forward_feature_test[num]
    x_reverse_feature_test_bal = x_reverse_feature_test[num]
    y_forward_feature_test_bal = y_forward_feature_test[num]
    y_reverse_feature_test_bal = y_reverse_feature_test[num]

    print(label_train.shape)
    print(label_val.shape)
    print(label_test.shape)
    print(label_test_bal.shape)
    # 模型构建
    # filepath = "model.pickle"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
    cnn = model.IChrom_seq(input_shape_x_forward, input_shape_x_reverse, input_shape_y_forward, input_shape_y_reverse)
    cnn.compile(loss=[model.binary_focal_loss(alpha=0.9, gamma=2)], optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    # 0.9和2目前效果最好
    # cnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    cnn.fit(x=[x_forward_feature_train, x_reverse_feature_train, y_forward_feature_train, y_reverse_feature_train],
            y=label_train, batch_size=BATCH_SIZE,
            epochs=MAX_EPOCH, validation_data=(
            [x_forward_feature_val, x_reverse_feature_val, y_forward_feature_val, y_reverse_feature_val], label_val),
            callbacks=[early_stopping_monitor])
    # cnn = load_model(filepath)

    model_pro = cnn.predict(
        [x_forward_feature_test, x_reverse_feature_test, y_forward_feature_test, y_reverse_feature_test])
    model_pro_bal = cnn.predict([x_forward_feature_test_bal, x_reverse_feature_test_bal, y_forward_feature_test_bal,
                                 y_reverse_feature_test_bal])
    model_class = np.around(model_pro)
    model_class_bal = np.around(model_pro_bal)

    # Evaluation
    auprc_bal[i] = average_precision_score(label_test_bal, model_pro_bal)
    acc_bal[i] = accuracy_score(label_test_bal, model_class_bal)
    mcc_bal[i] = matthews_corrcoef(label_test_bal, model_class_bal)
    precision_bal[i], recall_bal[i], f1_bal[i], _ = precision_recall_fscore_support(label_test_bal, model_class_bal,
                                                                                    average='binary')
    print('auprc_bal:', auprc_bal)
    print('acc_bal:', acc_bal)
    print('mcc_bal:', mcc_bal)
    print('precision_bal:', precision_bal)
    print('recall_bal', recall_bal)
    print('f1_bal:', f1_bal)
    print(auprc_bal[i], acc_bal[i], mcc_bal[i], precision_bal[i], recall_bal[i], f1_bal[i])
    print(len(index_test))

    i += 1

print('10-fold cross-validation')
print('auprc_bal:', auprc_bal.mean())
print('acc_bal:', acc_bal.mean())
print('mcc_bal:', mcc_bal.mean())
print('precision_bal:', precision_bal.mean())
print('recall_bal', recall_bal.mean())
print('f1_bal:', f1_bal.mean())
print(auprc_bal.mean(), acc_bal.mean(), mcc_bal.mean(), precision_bal.mean(), recall_bal.mean(), f1_bal.mean())
print('----------------------------------------------------------------')
for i in range(10):
    print(auprc_bal[i], acc_bal[i], mcc_bal[i], f1_bal[i])



