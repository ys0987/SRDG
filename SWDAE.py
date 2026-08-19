import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle
from swd_util import sw_loss,sw_loss2
def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)



def train_on_step(images_batch):
    # print(images_batch.shape[1])
    image_batch = tf.reshape(images_batch, shape=[-1, images_batch.shape[1], 1])
    loss=train_loss(image_batch)

    return loss

def train_loss(image_batch):
    with tf.GradientTape() as tape:
        # model
        recon_image = dcae(image_batch, training=True)
        #loss in sliced wasserstein

        # image_batch,recon_image=tf.squeeze(image_batch),tf.squeeze(recon_image)
        image_batch = tf.reshape(image_batch, [tf.shape(image_batch)[0], -1])
        recon_image = tf.reshape(recon_image, [tf.shape(recon_image)[0], -1])
        recon_loss = sw_loss2(true_distribution=image_batch,
                             generated_distribution=recon_image,
                             num_projections=Num_Pro)
        # diff = recon_image - image_batch
        # recon_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        # print('loss:',image_batch.shape,images_batch.shape,recon_image.shape)
        total_loss = recon_loss
    gradients = tape.gradient(total_loss, dcae.trainable_weights)
    opt.apply_gradients(zip(gradients, dcae.trainable_weights))

    return total_loss

#查看时间
import time
start_time = time.time()
# batch_size=32 #batch_size=32 (大样本)
batch_size=8 #batch_size=2 （小样本）
num_epochs=8000
learning_rate=2e-4 #learning_rate=2e-4
img_size=3072
img_channels=1
endim=256
interdim=256
Num_Pro=64
# Global Settings
rpm='1500'
fault_name='C0_train'
file_name='./split_'+rpm+'/'+fault_name+'.pkl'
with open(file_name, 'rb') as f:
    data_train = pickle.load(f)

    np.random.seed(23)
    # 随机选择 k个样本
    mix = [i for i in range(len(data_train))]
    np.random.shuffle(mix)
    train_x = data_train[mix]
    random_x = train_x[0:840, ]
    # ================= 增加归一化预处理 =================
    # 计算每个样本的均值和标准差，进行 Z-score 标准化
    # 这样处理后，信号的均值为 0，标准差为 1
    mean = np.mean(random_x, axis=1, keepdims=True)
    std = np.std(random_x, axis=1, keepdims=True)
    random_x = (random_x - mean) / (std + 1e-8)  # 加上 1e-8 防止除以 0

    print(f"归一化后 - 最大值: {random_x.max():.4f}, 最小值: {random_x.min():.4f}")
    # =================================================
data_train=tf.cast(random_x,dtype=tf.float32)
train_ds=tf.data.Dataset.from_tensor_slices(data_train).shuffle(10000).batch(batch_size)

# #inputs
inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
# # 2，神经网络
layers=tf.keras.layers
# # ### Encoder
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
#
# # # #Build model
dcae=keras.Model(inputs_, rx)
#
# # # # Opimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
print('Network Summary-->')
dcae.summary()
#
#
# --------------->>>Training Phase<<<---------------------------
# Run
loss_list=[]
# total_batch=350
# total_batch = int(len(data_train) / batch_size)
# total_batch = int(np.ceil(len(data_train) / batch_size))
total_batch = tf.data.experimental.cardinality(train_ds).numpy()
for epoch in range(num_epochs):
    ave_cost=0
    for images_batch in train_ds:
        # print(images_batch.shape)
        loss=train_on_step(images_batch)
        ave_cost+=float(loss)/total_batch
    print("Epoch:", (epoch + 1), "cost =", ave_cost)
    loss_list.append(ave_cost)

    # Save the model weights every 100 epoches
    if epoch%1000==0:
      save_dir='./ae_results_per_speed/ae_model/1500/model'+fault_name+'/'
      os.makedirs(save_dir, exist_ok=True)
      dcae.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')

# save loss
loss_curve = np.array(loss_list)
save_loss = './ae_results_per_speed/ae_model/1500/loss/'
os.makedirs(save_loss, exist_ok=True)
np.savetxt(save_loss + 'ae_loss'+fault_name+'.txt', loss_curve)

# Save the model weights in the last step
save_dir='./ae_results_per_speed/ae_model/1500/model'+fault_name+'/'
dcae.save_weights(save_dir+'model_last_'+str(epoch)+'.ckpt')
print('Optimization Finished')
end_time = time.time()
print(f"代码运行时间: {end_time - start_time:.6f} 秒")

import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.show()