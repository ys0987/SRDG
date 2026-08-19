# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from pathlib import Path
from tqdm import tqdm  # 导入进度条库

# ==========================================
# 1. 手动修改
# ==========================================
rpm = '1500'
enc_name = 'AE_enc_C0_train'  #

# 路径配置 (自动转为绝对路径，防止 transformers 报错)
# 修复了路径中多余的单引号，并将基础路径退回到 ae_model 层级，方便后续正确拼接
BASE_PATH = os.path.abspath("./Results/ae_model")
MODEL_DIR = os.path.abspath("./vicuna-7b-v1.5")
SAVE_DIR = os.path.abspath("./Outputs_128")

# 维度配置
AE_DIM = 256
LLM_DIM = 4096
TARGET_DIM = 128

# 训练配置
AE_EPOCHS = 100
AE_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据加载
# ==========================================
def load_my_data():
    # 修复完整的路径拼接逻辑：BASE_PATH / rpm / "Encoded" / enc_name
    file_name = os.path.join(BASE_PATH, rpm, "Encoded", enc_name + '.pkl')
    print(f"\n[1/4] 正在加载数据: {file_name}")

    with open(file_name, 'rb') as f:
        x_all = pickle.load(f)
    x_all = np.array(x_all)
    return torch.from_numpy(x_all).float().to(DEVICE)


# ==========================================
# 3. 模型定义
# ==========================================
class LLMAdapter(nn.Module):
    def __init__(self, in_dim=256, out_dim=4096):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.fc(x).unsqueeze(1)


class AECompressor(nn.Module):
    def __init__(self, in_dim=4096, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024), nn.ReLU(),
            nn.Linear(1024, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 1. 加载数据
    x_tensor = load_my_data()

    # 2. 加载 Vicuna
    print(f"[2/4] 正在加载本地 LLM: {MODEL_DIR}")
    base_llm = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE)
    base_llm.eval()

    # 3. 初始化模块
    adapter = LLMAdapter(AE_DIM, LLM_DIM).to(DEVICE)
    ae = AECompressor(LLM_DIM, TARGET_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(ae.parameters()), lr=AE_LR)

    # 4. 训练对齐与重构 (加入进度条)
    print(f"[3/4] 开始训练类别 {enc_name} 的压缩网络...")

    # 使用 tqdm 包装循环
    pbar = tqdm(range(AE_EPOCHS), desc=f"Training {enc_name}")

    for ep in pbar:
        # AE -> Adapter -> LLM
        llm_input = adapter(x_tensor)
        with torch.no_grad():
            out = base_llm(inputs_embeds=llm_input.to(base_llm.dtype))
            h_llm = out.last_hidden_state.squeeze(1).float()

        # AE 压缩与重构
        z_128, recon = ae(h_llm)
        loss = F.mse_loss(recon, h_llm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条右侧的 Loss 信息
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    # 5. 保存结果
    print(f"[4/4] 正在保存结果...")
    adapter.eval()
    ae.eval()
    with torch.no_grad():
        final_h = base_llm(inputs_embeds=adapter(x_tensor).to(base_llm.dtype)).last_hidden_state.squeeze(1).float()
        final_z_128 = ae.encoder(final_h)

    out_path = Path(SAVE_DIR) / rpm
    out_path.mkdir(parents=True, exist_ok=True)

    # 保存特征和模型
    np.save(out_path / f"{enc_name}_target128.npy", final_z_128.cpu().numpy())
    torch.save(adapter.state_dict(), out_path / f"{enc_name}_adapter.pth")
    torch.save(ae.state_dict(), out_path / f"{enc_name}_ae.pth")

    print(f"\n>>> 处理完成！128维目标特征已存至: {out_path / f'{enc_name}_target128.npy'}")


if __name__ == "__main__":
    main()