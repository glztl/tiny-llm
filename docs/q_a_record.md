# Token

Token 是大模型能理解的最小文本单位，就像人类用“字”或“词”思考，大模型用 Token 思考。

## Token 的三种粒度（从粗到细）

### 字符级（Character-level）
- 文本示例：`Hello`
- Token 划分：`['H', 'e', 'l', 'l', 'o']`
- Token 数量：5 个
- 优点：
  1. 词表极小（英文 ~65 个字符，中文 ~3000 常用字）
  2. 不会出现 OOV 问题
  3. 实现简单，适合学习
- 缺点：
  1. 序列太长（如“人工智能”→6 个 Token），计算慢
  2. 模型需要更深层才能理解“词”的含义

### 词级（Word-level）
- 文本示例：`Hello world`
- Token 划分：`['Hello', 'world']`
- Token 数量：2 个
- 优点：
  1. 序列短，计算快
  2. 每个 Token 有完整语义
- 缺点：
  1. 词表巨大（英文 10 万+，中文 50 万+）
  2. 遇到新词（如“LLaMA-3”）无法处理 → OOV 问题
  3. 中文需要额外分词工具（结巴等）

### 子词级（Subword，工业界主流 BPE/WordPiece）
- 文本示例：`Hello LLaMA-3`
- Token 划分：`['Hello', 'LL', 'a', 'MA', '-3']`
- Token 数量：5 个
- 原理：
  1. 高频词保持完整：`'Hello'` → 1 个 Token
  2. 低频词拆分成字句：`'LLaMA-3'` → 多个 Token
  3. 平衡词表大小和序列长度
- 优点：
  1. 词表适中（3 ~ 10 万）
  2. 能处理新词（拆分成已知子词）
  3. 序列长度合理
- 代表算法：
  1. BPE（Byte Pair Encoding）：GPT 系列
  2. WordPiece：BERT, QWen
  3. SentencePiece：LLaMA，支持多种语言

---

# 训练和推理的区别

- **训练（Train）**：学习的过程（更新权重）
- **推理（Inference）**：使用的过程（生成结果）
- **预测（Prediction）**：核心任务（输出一个值）

---


# 什么是自注意力机制（Self-Attention）？它如何工作？
自注意力是一种让每个 Token 能关注序列中其他 Token 的机制，通过加权求和方式动态聚合信息。它能捕捉长距离依赖，是 Transformer 的核心。

# 多头注意力（Multi-Head Attention）有什么作用？为什么要多头？
多头注意力将注意力机制并行分成多组，每组学习不同的子空间特征，提升模型表达能力和稳定性。

# 什么是位置编码（Positional Encoding），为什么 Transformer 需要它？
位置编码为每个 Token 注入位置信息，弥补 Transformer 无法感知序列顺序的缺陷。常见有正弦余弦编码和可学习编码。

# BPE、WordPiece、SentencePiece 有什么区别？各自适用场景？
BPE 通过合并高频子串构建词表，WordPiece 用于概率最大化分词，SentencePiece 支持多语言和无空格文本。BPE 常用于 GPT，WordPiece 用于 BERT，SentencePiece 用于 LLaMA。

# 什么是 KV Cache？它如何加速推理？
KV Cache 缓存已生成 Token 的 Key/Value，避免重复计算历史内容，大幅提升自回归推理速度。

# 为什么要用 LayerNorm？它在 Transformer 中的位置有何讲究？
LayerNorm 归一化特征分布，提升训练稳定性。可放在残差前（Pre-LN）或后（Post-LN），Pre-LN 更易训练深层模型。

# 残差连接（Residual Connection）在深层网络中的作用是什么？
残差连接缓解梯度消失/爆炸，促进信息流动，使深层网络更易优化。

# 什么是 Masked Self-Attention？为什么要做掩码？
Masked Self-Attention 用于生成任务，掩盖未来 Token，防止信息泄漏，保证自回归生成。

# LLM 训练时常用哪些优化器？AdamW 和 Adam 有什么区别？
常用 AdamW、Adam。AdamW 将权重衰减与梯度分离，收敛更好，适合 Transformer。

# 什么是梯度裁剪（Gradient Clipping），为什么要用？
梯度裁剪限制梯度最大范数，防止梯度爆炸，提升训练稳定性。

# 如何实现混合精度训练（AMP）？带来哪些好处？
AMP 结合 float16 和 float32 训练，减少显存占用，加速训练，常用 PyTorch 的 autocast 实现。

# 什么是 LoRA 微调？它的原理和优势是什么？
LoRA 通过插入低秩矩阵微调大模型，仅训练少量参数，节省显存和计算，适合下游任务。

# 如何实现多 GPU 训练？常见的分布式训练策略有哪些？
可用 Data Parallel、Distributed Data Parallel（DDP）、模型并行等。DDP 是主流方案，效率高。

# LLM 推理时常用哪些采样方法？Top-K、Top-P 有什么区别？
常用 Greedy、Top-K、Top-P（Nucleus）采样。Top-K 固定选概率前 K 个，Top-P 动态选累计概率超过 P 的 Token。

# 如何理解“自回归生成”？与 Encoder-Decoder 架构有何不同？
自回归生成每次输出一个 Token，依赖历史输出。Encoder-Decoder 结构适合输入输出不对齐任务（如翻译）。

# 什么是 OOV 问题？各类分词方法如何应对？
OOV（Out-of-Vocabulary）指词表外词。子词分词（BPE 等）可将新词拆分为已知子词，缓解 OOV。

# 如何保存和加载 PyTorch 模型？state_dict 有什么作用？
用 torch.save(model.state_dict()) 保存，用 model.load_state_dict() 加载。state_dict 只包含参数权重，便于迁移和部署。

# LLM 量化部署的常见方法有哪些？GGUF 是什么？
常见有 int8、int4 量化，减少模型体积和推理成本。GGUF 是一种高效的量化模型格式，便于跨平台部署。

# 如何评估 LLM 的生成质量？常用指标有哪些？
常用 Perplexity、BLEU、ROUGE、人工评测等。不同任务选用不同指标。

# 大模型训练中常见的工程难点和优化点有哪些？
包括显存管理、分布式训练、数据管道、混合精度、梯度累积、模型并行等。

> 你还可以补充更多问题，或针对每个问题深入准备面试答案。

