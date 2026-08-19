# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import random
import warnings
from torch.utils.data import TensorDataset, DataLoader

# 屏蔽 NumPy 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ================= 1. 配置参数 =================
class Args:
    def __init__(self):
        # 两个源域对一个目标域
        self.source1_paths = [f'./Final_LSTM_Aligned/1500/AE_enc_C{i}_train_final.pkl' for i in range(5)]
        self.source2_paths = [f'./Final_LSTM_Aligned/2000/AE_enc_C{i}_train_final.pkl' for i in range(5)]
        self.target_paths = [f'./Final_LSTM_Aligned/2500/AE_enc_C{i}_test_final.pkl' for i in range(5)]

        self.input_dim = 128
        self.num_classes = 5

        self.epochs = 200
        self.batch_size = 64
        self.lr = 1e-3
        self.weight_decay = 1e-2

        # 🌟 CORAL Loss 的权重，控制“强制对齐”的力度
        self.coral_weight = 5.0


args = Args()


# ================= 2. 数据加载与处理 =================
def load_features(paths_list):
    X_list, Y_list = [], []
    for paths in paths_list:
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                feat = np.array(pickle.load(f)).astype(np.float32)
                if len(feat.shape) == 3:
                    feat = feat.reshape(feat.shape[0], -1)

                label = -1
                for class_idx in range(args.num_classes):
                    if f"_C{class_idx}_" in p:
                        label = class_idx
                        break
                if label != -1:
                    X_list.append(feat)
                    Y_list.append(np.full(len(feat), label))

    if not X_list:
        raise ValueError("❌ 未找到数据，请检查路径！")

    return np.vstack(X_list), np.concatenate(Y_list).astype(np.int64)


# ================= 3. 核心网络组件与 Loss =================
class TemperatureCrossEntropy(nn.Module):
    def __init__(self, temperature=2.0, label_smoothing=0.1):
        super().__init__()
        self.T = temperature
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        return self.ce(logits / self.T, labels)


def coral_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    tmp_s = torch.ones((1, ns)).to(source.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    tmp_t = torch.ones((1, nt)).to(target.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits


# ================= 4. Trainer 训练器 =================
class CORALTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" 使用设备: {self.device}")

        self.model = MLPClassifier(input_dim=args.input_dim, num_classes=args.num_classes).to(self.device)
        self._prepare_dataloaders()

        self.criterion_cls = TemperatureCrossEntropy(temperature=2.0, label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-5)


    def _prepare_dataloaders(self):
        print("\n 正在加载并处理数据...")

        X_src1, Y_src1 = load_features([self.args.source1_paths])
        X_src2, Y_src2 = load_features([self.args.source2_paths])

        X_src_all = np.vstack((X_src1, X_src2))

        self.mean = X_src_all.mean(axis=0)
        self.std = X_src_all.std(axis=0) + 1e-8

        X_src1 = (X_src1 - self.mean) / self.std
        X_src2 = (X_src2 - self.mean) / self.std

        X_tgt, Y_tgt = load_features([self.args.target_paths])
        X_tgt = (X_tgt - self.mean) / self.std

        self.loader_src1 = DataLoader(TensorDataset(torch.FloatTensor(X_src1), torch.LongTensor(Y_src1)),
                                      batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.loader_src2 = DataLoader(TensorDataset(torch.FloatTensor(X_src2), torch.LongTensor(Y_src2)),
                                      batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.target_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tgt), torch.LongTensor(Y_tgt)),
                                        batch_size=self.args.batch_size, shuffle=False)

    def train_epoch(self):
        self.model.train()
        total_loss_val = 0
        correct, total = 0, 0

        for (data1, label1), (data2, label2) in zip(self.loader_src1, self.loader_src2):
            data1, label1 = data1.to(self.device), label1.to(self.device)
            data2, label2 = data2.to(self.device), label2.to(self.device)

            feat1, logits1 = self.model(data1)
            feat2, logits2 = self.model(data2)

            loss_cls = self.criterion_cls(logits1, label1) + self.criterion_cls(logits2, label2)
            loss_coral = coral_loss(feat1, feat2)

            loss = loss_cls + self.args.coral_weight * loss_coral

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss_val += loss.item()

            _, pred = logits1.max(1)
            total += label1.size(0)
            correct += pred.eq(label1).sum().item()

        return total_loss_val / len(self.loader_src1), 100. * correct / total

    def evaluate(self, loader, return_labels=False):
        self.model.eval()
        total_loss = 0
        correct, total = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(self.device), label.to(self.device)
                _, logits = self.model(data)

                loss = self.criterion_cls(logits, label)
                total_loss += loss.item()

                _, pred = logits.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

                if return_labels:
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total

        if return_labels:
            return avg_loss, acc, np.array(all_labels), np.array(all_preds)
        return avg_loss, acc


    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            train_loss_val, train_acc = self.train_epoch()
            self.scheduler.step()

            obs_target_loss, obs_target_acc = self.evaluate(self.target_loader)

            if epoch == 1 or epoch % 10 == 0 or epoch == self.args.epochs:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"   [Epoch {epoch:03d}] LR: {current_lr:.6f} | Train Loss: {train_loss_val:.4f} | Test Loss: {obs_target_loss:.4f} | Src Acc: {train_acc:.2f}% | 👁️ Tgt Acc: {obs_target_acc:.2f}%")

        _, final_target_acc, y_true, y_pred = self.evaluate(self.target_loader, return_labels=True)

        return final_target_acc


# ================= 5. Main 流程控制 =================
def set_random_seed(seed):
    """设置全局随机种子以保证当前轮次可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    seed = 42
    set_random_seed(seed)

    print("=" * 65)
    print(f"🚀 开始进行 Deep CORAL 训练 (Seed: {seed})")
    print("=" * 65)

    trainer = CORALTrainer(args)
    target_acc = trainer.train()

    print("\n" + "🏆" * 15)
    print(f"✅ 实验结束，目标域最终准确率: {target_acc:.2f}%")
    print("🏆" * 15)


if __name__ == "__main__":
    main()