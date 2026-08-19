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

请勿提交无公开授权的数据。大规模数据建议存放在 Git LFS、Zenodo、Figshare 或其他数据托管平台，并在 README 中提供下载链接及校验值。

## 本地模型准备

`llm.py` 默认从项目根目录下的 `vicuna-7b-v1.5` 加载本地模型：

```text
vicuna-7b-v1.5/
|-- config.json
|-- tokenizer_config.json
|-- tokenizer.model
`-- model-*.safetensors
```

模型文件体积较大，不应直接提交到 GitHub。请根据模型发布方的许可证自行获取模型，并确保使用方式符合其许可要求。如模型位于其他位置，请修改 `llm.py` 中的 `MODEL_DIR`。

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

该步骤需要较多显存。输出的 128 维特征、适配器权重和自编码器权重会保存到 `Outputs_128/`。

### 4. 训练 LSTM 对齐网络

在 `lstm.py` 中设置 `TARGET_RPM`，确认 `Results/ae_model/` 与 `Outputs_128/` 中已包含该转速下 `C0` 至 `C4` 的训练数据，然后运行：

```bash
python lstm.py
```

对齐后的训练和测试特征会保存到 `Final_LSTM_Aligned/`。

### 5. 运行 Deep CORAL 分类

确认 `DEEPcoral.py` 中的两个源域和目标域路径与实验设置一致，然后运行：

```bash
python DEEPcoral.py
```

默认配置使用 1500 RPM 和 2000 RPM 作为源域、2500 RPM 作为目标域。

## 可复现性

`DEEPcoral.py` 默认使用随机种子 `42`。其他脚本尚未统一固定 Python、NumPy、TensorFlow 和 PyTorch 的随机种子，因此不同运行之间的结果可能存在差异。正式发布实验结果前，建议统一随机种子并记录以下信息：

- Python、TensorFlow、PyTorch 和 CUDA 版本；
- GPU 型号和显存；
- 每个转速、类别的样本数量；
- 数据划分方式；
- 所有超参数与随机种子；
- 多次独立实验的均值和标准差。

```gitignore
.idea/
__pycache__/
*.py[cod]
.venv/
venv/

vicuna-7b-v1.5/
ae_results_per_speed/
Results/
Outputs_128/
Final_LSTM_Aligned/

*.ckpt*
*.h5
*.pth
*.pt
*.safetensors
```

## 引用

如果本项目用于论文，请在代码公开后补充论文引用信息：

```bibtex
@article{your_reference,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {Year}
}
```
