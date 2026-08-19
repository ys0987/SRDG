# SWDAE-LLM 跨转速故障诊断

本项目实现了一套面向跨转速机械故障诊断的实验流程。整体方法首先使用切片 Wasserstein 自编码器（SWDAE）提取信号特征，再通过本地大语言模型生成目标表征，随后使用 LSTM 完成特征对齐，最后利用 Deep CORAL 进行跨域分类。

> 本仓库当前为研究代码。部分训练参数、转速和类别需要在脚本顶部手动修改，运行前请先阅读“已知限制”部分。

## 方法流程

```text
原始振动信号（3072 维）
        |
        v
SWDAE 训练与特征提取（256 维）
        |
        v
本地 Vicuna 模型 + 压缩网络（128 维目标表征）
        |
        v
LSTM 特征对齐（128 维）
        |
        
        v
Deep CORAL 跨转速分类
```

## 文件说明

```text
.
|-- SWDAE.py          # 训练切片 Wasserstein 自编码器
|-- encoder_AE.py     # 加载 SWDAE 权重并提取训练/测试特征
|-- swd_util.py       # Sliced Wasserstein Distance 损失函数
|-- llm.py            # 使用本地 Vicuna 模型生成并压缩目标表征
|-- lstm.py           # 使用 LSTM 对齐 SWDAE 特征与 LLM 目标表征
|-- DEEPcoral.py      # Deep CORAL 跨域分类
|-- split_1500/       # 1500 RPM 原始训练/测试数据
|-- split_2000/       # 2000 RPM 原始训练/测试数据
`-- split_2500/       # 2500 RPM 原始训练/测试数据
```

训练过程中还会使用或生成以下目录：

```text
ae_results_per_speed/ # SWDAE 模型权重和损失记录
Results/ae_model/     # SWDAE 编码及重构结果
vicuna-7b-v1.5/       # 本地大语言模型（需自行获取）
Outputs_128/          # LLM 目标表征及压缩模型
Final_LSTM_Aligned/   # LSTM 对齐后的特征及权重
```

## 环境要求

- Python 3.9 或 3.10
- 支持 CUDA 的 NVIDIA GPU（推荐）
- TensorFlow 2.x
- PyTorch 2.x
- Transformers

安装主要依赖：

```bash
pip install numpy tensorflow torch transformers tqdm matplotlib keras
```

为保证结果可复现，建议根据实际运行环境生成带版本号的依赖文件：

```bash
pip freeze > requirements.txt
```

TensorFlow、PyTorch 与 CUDA 的版本必须相互兼容，具体安装方式请参考各框架官方文档。

## 数据准备

```
每个 `.pkl` 文件应包含一个形状为 `[样本数, 3072]` 的 NumPy 数组或可转换为该形状的数组。代码默认包含 5 个类别，即 `C0` 至 `C4`。

## 本地模型准备

`llm.py` 默认从项目根目录下的 `vicuna-7b-v1.5` 加载本地模型：

```text
vicuna-7b-v1.5/
|-- config.json
|-- tokenizer_config.json
|-- tokenizer.model
`-- model-*.safetensors
```
## 运行步骤

所有命令都应在项目根目录执行。

### 1. 训练 SWDAE

在 `SWDAE.py` 中设置 `rpm` 和 `fault_name`，然后运行：

```bash
python SWDAE.py
```

需要针对每个转速和类别分别训练。模型权重会保存到 `ae_results_per_speed/`。

### 2. 提取 SWDAE 特征

在 `encoder_AE.py` 中设置 `rpm`、训练/测试文件名以及待加载的权重路径，然后运行：

```bash
python encoder_AE.py
```

编码特征和重构结果会保存到 `Results/ae_model/`。

### 3. 生成 LLM 目标表征

在 `llm.py` 中设置 `rpm`、`enc_name` 和 `MODEL_DIR`，然后运行：

```bash
python llm.py
```

### 4. 训练 LSTM 对齐网络

在 `lstm.py` 中设置 `TARGET_RPM`，确认 `Results/ae_model/` 与 `Outputs_128/` 中已包含该转速下 `C0` 至 `C4` 的训练数据，然后运行：

```bash
python lstm.py
```

### 5. 运行 Deep CORAL 分类

确认 `DEEPcoral.py` 中的两个源域和目标域路径与实验设置一致，然后运行：

```bash
python DEEPcoral.py
```

使用两个作为源域、一个作为目标域。


