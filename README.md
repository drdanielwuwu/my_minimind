# MiniMind - 轻量级语言模型框架

MiniMind 是一个轻量级的语言模型框架，专注于提供高效、灵活的Transformer架构实现，支持基础模型和MoE（Mixture of Experts）模型。

## 项目结构

```
minimind/
├── dataset/             # 数据集相关代码
│   ├── lm_dataset.py    # 语言模型数据集
│   └── pretrain_t2t_mini.jsonl  # 预训练数据
├── model/               # 模型定义
│   ├── MokioModel.py    # 核心模型实现
│   ├── tokenizer.json   # 分词器配置
│   └── tokenizer_config.json  # 分词器配置
├── out/                 # 模型输出目录
│   └── pretrain_512_gpu.pth  # 预训练模型
├── trainer/             # 训练相关工具
│   └── trainer_utils.py # 训练工具函数
├── test_gpu.py          # GPU测试脚本
├── test_gpu0.py         # 另一个GPU测试脚本
└── train_pretrain_gpu.py  # GPU预训练脚本
```

## 核心特性

- **基础Transformer架构**：实现了标准的Transformer编码器-解码器结构
- **MoE支持**：集成了Mixture of Experts架构，提高模型容量和效率
- **YaRN RoPE扩展**：支持长序列建模，通过YaRN方法扩展上下文长度
- **Flash Attention**：利用PyTorch的Flash Attention加速注意力计算
- **混合精度训练**：支持bfloat16和float16混合精度训练，提高训练速度和减少内存使用
- **梯度累积**：支持梯度累积，允许在有限GPU内存下使用更大的批量大小
- **KV Cache**：实现了KV缓存，加速推理过程

## 模型配置

MiniMind模型支持以下关键配置参数：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| hidden_size | 512 | 模型隐藏层维度 |
| num_hidden_layers | 8 | 模型层数 |
| num_attention_heads | 8 | 注意力头数 |
| num_key_value_heads | 2 | 键值注意力头数 |
| vocab_size | 6400 | 词汇表大小 |
| max_position_embeddings | 32768 | 最大位置编码长度 |
| use_moe | False | 是否使用MoE架构 |
| num_experts_per_tok | 2 | 每个token使用的专家数 |
| n_routed_experts | 4 | 路由专家数量 |
| n_shared_experts | 1 | 共享专家数量 |

## 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd minimind

# 安装依赖
pip install torch transformers tqdm
```

## 快速开始

### 预训练模型

使用以下命令启动预训练：

```bash
python train_pretrain_gpu.py
```

预训练配置可在脚本中修改，主要参数包括：

- `batch_size`：批量大小（默认32）
- `learning_rate`：学习率（默认5e-4）
- `epochs`：训练轮数（默认1）
- `hidden_size`：模型隐藏层维度（默认512）
- `num_hidden_layers`：模型层数（默认8）
- `max_seq_len`：最大序列长度（默认512）
- `use_moe`：是否使用MoE架构（默认False）
- `device`：设备选择（自动检测，优先使用GPU）
- `dtype`：数据类型（默认bfloat16）

### 测试模型

使用以下命令测试模型：

```bash
python test_gpu.py
```

测试脚本会执行以下操作：
1. 检查GPU是否可用，自动切换到CPU模式（如果需要）
2. 加载模型配置和初始化模型
3. 尝试加载训练好的权重文件（如果存在）
4. 测试文本生成功能
5. 测试批量推理性能
6. 测试混合精度推理速度

测试输入示例：
```
测试输入: 人工智能是
生成文本: 人工智能是一种模拟人类智能的技术，它可以通过机器学习、深度学习等方法来实现。人工智能的应用非常广泛，包括图像识别、语音识别、自然语言处理等领域。
```

## 数据格式

预训练数据使用JSONL格式，每行包含一个训练样本：

```json
{"text": "这里是训练文本..."}
```

数据集加载器会自动处理数据清洗，包括：
- 跳过空行和无效数据
- 处理编码错误
- 确保文本数据的完整性

## 训练工具

`trainer/trainer_utils.py` 提供了以下工具函数和类：

- `is_main_process()`：检查是否是主进程（用于分布式训练）
- `Logger()`：日志记录函数，只在主进程输出
- `get_lr()`：学习率调度函数，实现了余弦退火策略
- `init_distributed_mode()`：初始化分布式训练模式
- `setup_seed()`：设置随机种子，确保实验可重复性
- `lm_checkpoint()`：保存和加载模型检查点
- `init_model()`：初始化模型和分词器，支持加载预训练权重
- `SkipBatchSampler`：批采样器，支持跳过指定数量的批次

### 学习率调度

使用余弦退火策略，学习率从初始值逐渐衰减到初始值的0.1倍：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### 分布式训练支持

提供了分布式训练的初始化函数，可以在多GPU环境中使用：

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 模型检查点

支持保存和加载模型检查点，包括模型权重、优化器状态、训练步数等信息：

```python
def lm_checkpoint(lm_config, weight="full_sft", model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir="checkpoints", **kwargs):
    # 保存或加载模型检查点
```

### 批采样器

支持跳过指定数量的批次，用于从中间开始训练：

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        # 初始化批采样器
    
    def __iter__(self):
        # 生成批次，跳过指定数量的批次
    
    def __len__(self):
        # 返回剩余的批次数
```


## 模型输出

训练完成后，模型会保存在`out/`目录下，文件格式为`pretrain_{hidden_size}_gpu.pth`。

## 技术细节

### 注意力机制

实现了标准的多头注意力机制，并支持Flash Attention加速。注意力计算采用pre-norm架构，即先对输入进行层归一化，再进行注意力计算。

### MoE实现

MoE（Mixture of Experts）实现包括：
- 门控机制：使用可学习的门控网络为每个token选择专家
- 专家路由：根据门控网络的输出，将token分配给不同的专家
- 加权融合：将不同专家的输出根据门控权重进行融合
- 辅助损失：添加辅助损失以平衡专家的使用

### RoPE位置编码

实现了YaRN（Yet Another RoPE Extension）方法，通过频率缩放扩展模型的上下文长度。

## 性能优化

- **混合精度训练**：使用bfloat16或float16减少内存使用和加速计算
- **梯度累积**：通过累积梯度实现更大的有效批量大小
- **KV Cache**：在推理过程中缓存键值对，减少重复计算
- **Flash Attention**：利用PyTorch的Flash Attention实现，加速注意力计算

## 未来计划

- [ ] 支持更多预训练任务
- [ ] 实现模型量化
- [ ] 支持分布式训练
- [ ] 提供更多模型大小的预训练权重
- [ ] 添加评估脚本和基准测试

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request，共同改进MiniMind框架。

## 联系方式

如有问题或建议，请通过以下方式联系：

- Email: [your-email@example.com]
- GitHub: [your-github-username]
