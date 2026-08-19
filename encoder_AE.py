import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle
from swd_util import sw_loss

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def train_on_step(images_batch):
    image_batch = tf.reshape(images_batch, shape=[-1, 3072, 1])
    loss=train_loss(image_batch)

    return loss

def train_loss(image_batch):
    with tf.GradientTape() as tape:
        # model
        recon_image = dcae(image_batch, training=True)
        #loss in sliced wasserstein
        image_batch,recon_image=tf.squeeze(image_batch),tf.squeeze(recon_image)
        recon_loss = sw_loss(true_distribution=image_batch,
                             generated_distribution=recon_image,
                             num_projections=image_batch.shape[0],
                             batch_size=image_batch.shape[0])

        # print('loss:',image_batch.shape,images_batch.shape,recon_image.shape)
        total_loss = recon_loss
    gradients = tape.gradient(total_loss, dcae.trainable_weights)
    opt.apply_gradients(zip(gradients, dcae.trainable_weights))

    return total_loss.numpy()

# Global Settings
batch_size=32 #batch_size=2
num_epochs=4000
learning_rate=2e-4 #learning_rate=2e-4
img_size=3072
img_channels=1
endim=128
interdim=256

#inputs
inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
# 2，神经网络
layers=tf.keras.layers
# ### Encoder
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(inputs_)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
print(x.shape)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x) #-1,16,64

x = layers.Flatten()(x) #3072

latent=layers.Dense(units=interdim, activation=leaky_relu)(x)

enc=layers.Dense(units=3072, activation='linear')(latent) #3072
#
x = tf.reshape(enc, [-1, 48, 64])
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)

x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)

x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)

x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)

x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)

x = layers.UpSampling1D(2)(x)
rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation='linear')(x)

# # # #Build model
dcae=keras.Model(inputs_, rx)
#
# # # # Opimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
print('Network Summary-->')
dcae.summary()

#data_test
rpm='1500'
#train
enc_name='C0_train'
file_name = './split_'+rpm+'/'+enc_name+'.pkl'
x = pickle.load(open(file_name, 'rb'))

# --- 加入归一化 ---
x_mean = np.mean(x, axis=1, keepdims=True)
x_std = np.std(x, axis=1, keepdims=True)
x = (x - x_mean) / (x_std + 1e-8)
data = tf.reshape(x, shape=[-1, 3072, 1])

#test
enc_name_test='C0_test'
file_name = './split_'+rpm+'/'+enc_name_test+'.pkl'
x2 = pickle.load(open(file_name, 'rb'))
# --- 加入归一化 ---
x2_mean = np.mean(x2, axis=1, keepdims=True)
x2_std = np.std(x2, axis=1, keepdims=True)
x2 = (x2 - x2_mean) / (x2_std + 1e-8)
data_test = tf.reshape(x2, shape=[-1, 3072, 1])

#加载AE.py训练的权重
dir_loadweights='./ae_results_per_speed/ae_model/1500/model'+enc_name+'/model_7999.ckpt'
print('Load weights from',dir_loadweights)
dcae.load_weights(dir_loadweights)

#进行数据映射模型建模，new_enout为中间特征映射模型，3072到256维度，new_enout2为重构数据模型，和输入数据一样为3072维度到3072
new_enout = tf.keras.models.Model(inputs=inputs_, outputs=latent)
new_enout2 = tf.keras.models.Model(inputs=inputs_, outputs=rx)

#进行数据映射
extracted_features = new_enout.predict(data)
reconstruct_data = new_enout2.predict(data).reshape(-1, img_size)
print(extracted_features.shape,reconstruct_data.shape)

extracted_features_test = new_enout.predict(data_test)
reconstruct_data_test = new_enout2.predict(data_test).reshape(-1, img_size)
print(extracted_features_test.shape,reconstruct_data_test.shape)

#保存映射后的数据
with open('./Results/ae_model/1500/Encoded/AE_enc_'+enc_name+'.pkl', 'wb') as f:
    pickle.dump(extracted_features, f, pickle.HIGHEST_PROTOCOL)
with open('./Results/ae_model/1500/Reconstruct/AE_re_'+enc_name+'.pkl', 'wb') as f:
    pickle.dump(reconstruct_data, f, pickle.HIGHEST_PROTOCOL)

with open('./Results/ae_model/1500/Encoded/AE_enc_' + enc_name_test + '.pkl', 'wb') as f:
    pickle.dump(extracted_features_test, f, pickle.HIGHEST_PROTOCOL)
with open('./Results/ae_model/1500/Reconstruct/AE_enc_' + enc_name_test + '.pkl', 'wb') as f:
    pickle.dump(reconstruct_data_test, f, pickle.HIGHEST_PROTOCOL)