import os
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import (Input, LSTM, Dense, Softmax, Conv1D, LayerNormalization, Dropout)
from keras.models import Model

# ==================== 1. 路径与配置 ====================
# 🌟 修正了基础路径，避免拼接时出现两个 1500
BASE_WAE_DIR = os.path.abspath("./Results/ae_model")
BASE_LLM_DIR = os.path.abspath("./Outputs_128")
BASE_SAVE_DIR = os.path.abspath("./Final_LSTM_Aligned")

# 🌟 指定单一转速
TARGET_RPM = '1500'

NUM_CLASSES = 5
TARGET_DIM = 128
EPOCHS = 100
BATCH_SIZE = 64


# ==================== 2. 增强型联合损失函数 ====================
class EnhancedAlignedLoss(tf.keras.losses.Loss):
    def __init__(self, tau=0.1, alpha=0.8):
        super().__init__()
        self.tau = tau
        self.alpha = alpha  # 权衡 对比损失(分离度) 和 MSE损失(LLM目标对齐)

    def call(self, y_true_combined, y_pred):
        # 分离目标特征和标签
        target_feat = y_true_combined[:, :128]
        labels = tf.cast(y_true_combined[:, 128:], tf.int32)
        labels = tf.reshape(labels, (-1,))

        # 1. MSE 对齐损失
        mse_loss = tf.reduce_mean(tf.square(y_pred - target_feat))

        # 2. 监督对比损失 (SupCon)
        z = tf.math.l2_normalize(y_pred, axis=-1)
        logits = tf.matmul(z, z, transpose_b=True) / self.tau

        mask = tf.cast(tf.equal(labels[:, None], labels[None, :]), tf.float32)
        batch_size = tf.shape(labels)[0]
        logits_mask = tf.ones((batch_size, batch_size)) - tf.eye(batch_size)
        mask = mask * logits_mask

        exp_logits = tf.exp(logits - tf.reduce_max(logits, axis=1, keepdims=True)) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)

        mask_sum = tf.reduce_sum(mask, axis=1)
        mask_sum = tf.where(mask_sum > 0, mask_sum, tf.ones_like(mask_sum))
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_sum

        supcon_loss = -tf.reduce_mean(mean_log_prob_pos)

        # 3. 联合损失
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * supcon_loss
        return total_loss


# ==================== 3. 增强型 LSTM 模型构建 ====================
def build_lstm_model():
    inputs = Input(shape=(1, 256))

    x = Conv1D(128, 1, padding="same", activation="gelu")(inputs)
    x = LayerNormalization()(x)
    x = LSTM(256, return_sequences=True)(x)

    att_w = Softmax(axis=1)(Dense(1, activation='tanh')(x))
    context = tf.reduce_sum(x * att_w, axis=1)

    x = Dense(256, activation='gelu')(context)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(128)(x)
    return Model(inputs, outputs)


# ==================== 4. 核心执行逻辑 ====================
rpm = TARGET_RPM
print(f"\n{'=' * 60}\n 启动处理转速: {rpm} RPM\n{'=' * 60}")

# --- A. 加载训练数据 ---
x_train_all, y_llm_all, l_train_all = [], [], []
for i in range(NUM_CLASSES):
    #
    wae_p = os.path.join(BASE_WAE_DIR, rpm, "Encoded", f"AE_enc_C{i}_train.pkl")
    llm_p = os.path.join(BASE_LLM_DIR, rpm, f"AE_enc_C{i}_train_target128.npy")

    if os.path.exists(wae_p) and os.path.exists(llm_p):
        with open(wae_p, 'rb') as f:
            w_data = np.array(pickle.load(f))
        l_data = np.load(llm_p)
        x_train_all.append(w_data)
        y_llm_all.append(l_data)
        l_train_all.append(np.full(len(w_data), i))

if not x_train_all:
    raise ValueError(f"❌ 错误: 未找到 {rpm} RPM 的训练数据对，请检查数据路径。")

X_train = np.expand_dims(np.vstack(x_train_all), axis=1)
Y_train_combined = np.column_stack([np.vstack(y_llm_all), np.concatenate(l_train_all)])

# 打乱数据
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train_combined = Y_train_combined[indices]

# 创建保存目录
rpm_save_dir = os.path.join(BASE_SAVE_DIR, rpm)
os.makedirs(rpm_save_dir, exist_ok=True)

# --- B. 训练模型 ---
model = build_lstm_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=EnhancedAlignedLoss(tau=0.1, alpha=0.5))

print(f" 开始训练增强版 {rpm} 对齐网络 (设定轮数: {EPOCHS})...")

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

model.fit(
    X_train, Y_train_combined,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[lr_callback],
    verbose=1
)

# 🌟 保存模型权重
weights_save_path = os.path.join(rpm_save_dir, f"lstm_aligned_weights_{rpm}.h5")
model.save_weights(weights_save_path)
print(f" 模型权重已成功保存至: {weights_save_path}")

# --- C. 推理预测与结构化保存 ---
print(f" 正在导出 {rpm} 训练及测试特征...")

for mode in ["train", "test"]:
    for i in range(NUM_CLASSES):
        #
        wae_p = os.path.join(BASE_WAE_DIR, rpm, "Encoded", f"AE_enc_C{i}_{mode}.pkl")
        if os.path.exists(wae_p):
            with open(wae_p, 'rb') as f:
                w_data = np.array(pickle.load(f))

            p_feat = model.predict(np.expand_dims(w_data, axis=1), verbose=0)


            save_path = os.path.join(rpm_save_dir, f"AE_enc_C{i}_{mode}_final.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(p_feat, f)

print(f"\n 增强对齐任务处理完毕！相关文件及模型权重已保存至 {rpm_save_dir} 目录下。")