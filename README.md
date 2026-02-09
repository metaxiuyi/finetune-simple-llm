# LLM 微调入门项目 (Qwen2.5-1.5B + Ruozhiba)

本项目旨在帮助开发者快速掌握 LLM 微调的全流程，包括环境配置、数据准备、LoRA 微调以及推理测试。
项目基于 `Qwen/Qwen2.5-1.5B-Instruct` 模型和 `ruozhiba` 数据集。

## 目录结构

- `prepare_data.py`: 数据准备脚本，下载并格式化数据集。
- `train.py`: 微调脚本，使用 LoRA 技术进行训练。
- `inference.py`: 推理脚本，加载微调后的模型进行对话。
- `finetune_in_colab.ipynb`: Google Colab 专用运行笔记。
- `pyproject.toml`: 项目依赖配置（使用 uv 管理）。

## Google Colab 运行

本项目已适配 Google Colab，你可以直接点击下方按钮打开 Notebook 进行运行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/metaxiuyi/finetune-simple-llm/blob/main/finetune_in_colab.ipynb)

**运行步骤**：
1. 点击上方按钮打开 Notebook。
2. 在 Colab 菜单栏选择 **Runtime (修改)** -> **Change runtime type (更改运行时类型)** -> **T4 GPU** (或 A100)。
3. 依次点击运行单元格即可。

## 快速开始 (本地运行)

### 1. 环境准备

本项目使用 `uv` 进行依赖管理。请确保已安装 `uv`（[安装指南](https://github.com/astral-sh/uv)）。

```bash
# 初始化环境并安装依赖
uv sync

# 激活环境
source .venv/bin/activate
```

### 2. 数据准备

运行以下命令下载并处理 `ruozhiba` 数据集：

```bash
uv run prepare_data.py
```

运行成功后，会在当前目录下生成 `ruozhiba_formatted.jsonl` 文件。

### 3. 模型微调

运行微调脚本（建议在 GPU 环境下运行）：

```bash
uv run train.py
```

训练完成后，模型权重（Adapter）将保存在 `qwen_ruozhiba_finetuned` 目录下。
*注意：脚本默认配置为单卡 GPU 运行，显存占用较小（约 6-8GB）。如果显存不足，可以尝试减小 `train.py` 中的 `per_device_train_batch_size`。*

### 4. 模型推理

加载微调后的模型进行对话测试：

```bash
uv run inference.py
```

### 5. 进阶说明

- **LoRA 配置**: 在 `train.py` 中可以修改 `LoraConfig` 参数（如 rank, alpha）来调整微调效果。
- **量化训练**: 脚本默认启用了 4-bit 量化（QLoRA）以节省显存。如果显存充足，可以在 `train.py` 中移除 `quantization_config` 以获得更快的训练速度或更好的精度。
- **合并权重**: 实际部署时，通常会将 Adapter 合并回基座模型。可以使用 `peft` 的 `merge_and_unload()` 方法实现。

## 常见问题

- **下载慢**: 如果连接 Hugging Face 慢，可以使用 `HF_ENDPOINT=https://hf-mirror.com` 环境变量加速。
- **显存溢出 (OOM)**: 请减小 batch size 或 max_seq_length。
