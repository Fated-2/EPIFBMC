import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score

# data_shape = (300, 1)  # 输入大小
data_shape = (300,)  # input dimension
batch_size = 32  # batch
latent_dim = 64

input_data = keras.Input(shape=data_shape)
x = layers.Dense(64, activation='relu')(input_data)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Create encoder，注意此处已经使用了z
# encoder = keras.Model(input_data, [z_mean, z_log_var, z], name='encoder')
encoder = keras.Model(input_data, z, name='encoder')
encoder.summary()

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(32, activation='relu')(latent_inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(300, activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# instantiate VAE model
# outputs = decoder(encoder(input_data)[2])
outputs = decoder(encoder(input_data))
vae = keras.Model(input_data, outputs, name='vae_mlp')
vae.summary()

reconstruction_loss = keras.losses.binary_crossentropy(input_data, outputs)
# reconstruction_loss *= 300
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.mean(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(x=x_train_pos, y=None,
        shuffle=True,
        epochs=1000,
        batch_size=batch_size,
        validation_data=(x_test[0:2 * int(y_test.sum())], None))  # 训练模型

encoder.save('vae_model/' + cell_lines + '_encoder.h5')
decoder.save('vae_model/' + cell_lines + '_decoder.h5')

# frist, pls load data
