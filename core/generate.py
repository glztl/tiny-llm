"""
推理生成脚本

加载训练好的模型，进行文本生成
支持交互式对话和批量生成

    Q: 推理和训练的区别是什么?
    Q: 如何控制生成的多样性?
"""

import torch
from config.config import device, block_size
from core.model import GPT
from core.data import SimpleTokenizer


# 加载模型函数
def load_model(filepath: str):
    """
    加载训练好的模型检查点

    Returns:
        model: GPT 模型
        tokenizer: SimpleTokenizer

        Q: 为什么保存state_dict而不是整个模型
        A: state_dict 只保存参数，文件更小，加载更灵活，不依赖类定义

        Q: weights_only=False 的作用?
        A: PyTorch 2.0+ 的安全选项，False允许加载任意对象 (需要信任文件来源)
    """
    print(f" 加载模型: {filepath}")

    """
    -filepath: 模型文件路径, 如"best_model.pth"
    -map_location=device: 将模型加载到指定设备 (如"cuda"或"cpu")
    -weights_only=False: 允许加载非张量对象 (如字典、列表)

    返回值 checkpoint 是一个字典，结构如下:
    {
        'model_state_dict': {...},     # 模型参数
        'vocab_size': 65,              # 词表大小
        'stoi': {...},                 # 字符到ID的映射
        'itos': {...},                 # ID到字符的映射
        'config': {...}                # 其他训练配置 (可选)
    }
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    """
    重加分词器 (Tokenizer)

    创建一个 "空" 的分词器实例
        Q: 为什么传空字符串? 
        A: 因为我们不需要构建词表, 会直接赋值

    直接恢复保存的映射关系
    这样分词器就能正确编码/解码文本了
        Q: 为什么要保存分词器?
        A: 不同训练数据的词表可能不同，必须保持一致才能正确解码
    """
    tokenizer = SimpleTokenizer("")
    tokenizer.string_to_int = checkpoint['stoi']
    tokenizer.int_to_string = checkpoint['itos']
    tokenizer.vocab_size = checkpoint['vocab_size']

    """
    重建模型架构

    先创建模型实例，再加载参数
    因为load_state_dict 只加载参数，不创建架构

        Q: 为什么需要 vocab_size?
        A: Embedding 层和 LM Head 的维度依赖词表大小
    """
    model = GPT(checkpoint['vocab_size'])

    """
    加载模型参数
    核心操作: 将保存的参数权重加载到模型中

    model_state_dict 是一个有序字典:
    {
        "token_embedding.table.weight": tensor([...]),
        "blocks.0.attn.key.weight": tensor([...]),
        "blocks.0.attn.query.weight": tensor([...]),
        ...

            Q: load_state_dict 和直接赋值有什么区别?
            A: load_state_dict 会验证参数名称和形状匹配，更安全
    }

    """
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # 设置为评估模式

    print(f" 模型加载成功")
    print(f" 词表大小: {tokenizer.vocab_size}")
    print(f" 设备: {device}")

    return model, tokenizer


# 生成函数 (带温度控制)
from typing import Optional

def generate_text(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_K: Optional[int] = None,
        seed: Optional[int] = None
):
    """
    生成文本

    Args:
        prompt: 输入提示
        max_new_tokens: 生成的新token数量
        temperature: 温度参数 (>1更随机, <1更确定)
        top_K: 采样，只从概率最高的 K 个词中采样
        seed: 随机种子，用于浮现结果
    """
    # 设置随机种子
    if seed is not None:
        # 保证每次运行生成相同结果，便于调试和浮现
        torch.manual_seed(seed)
    
    # 编码提示
    """
    假设prompt = "ROMEO:"
    编码后: [[45, 52, 49, 43, 52, 37]] 6个字符的ID

    形状变化:
        输入: "ROMEO:" (字符串 6个字符)
        输出: start[1, 6] [Batch=1, SeqLen=6]

        Q: 为什么是二维[1, 6]，而不是一维[6]?
        A: 模型期望输入是批量形式，即使只有一个样本，也需要保持批量维度
    """
    start = torch.tensor(
        [[tokenizer.string_to_int[c] for c in prompt]],
        dtype=torch.long,
        device=device
    )

    # 生成
    # 设置为评估模式，关闭Dropout，保证推理结果确定性
    model.eval()
    """
    禁用梯度追踪
    生成过程中不需要计算梯度，节省内存和计算资源
        Q: 推理时为什么不需要梯度?
        A: 推理不需要反向传播更新权重，只需要向前传播
    """
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截取上下文
            """
            第一次循环:
                输入: start[1, 6]
                输出: idx_cond[1, 6] (6 < 256， 全部保留)
            
            第100次循环:
                输入: start[1, 106]
                输出: idx_cond[1, 106] (106 < 256， 全部保留)

            第300次循环:
                输入: start[1, 306]
                输出: idx_cond[1, 256] (306 > 256,截取最后256个)

                Q: 为什么要截取?
                A: 位置编码只定义了 256 个位置，超出后会报错
            """
            idx_cond = start[:, -block_size:]

            # 前向传播 (获取预测分数)
            """
            输入: idx_cond [1, T_cond]
            输出: logits [1, T_cond, 65]

            物理意义:
                每个位置都有65个分数，代表"下一个词是各词的可能性"
                例如位置 6 的logits: [2.1, -0.5, 3.8, ..., 1.2]
            """
            logits, _ = model(idx_cond, None)
            """
            只取最后一个时间步的预测

            输入: logits [1, T_cond, 65]
            输出: logits [1, 65]

                Q: 为什么只取最后一个?
                A: 我们关心"下一个token"，不需要前面位置的预测
                    例如: "ROMEO:" 只需预测第 7 个字符，不需要预测 1-6 个
                示例：
                    原始 logits[0]: [[2.1, -0.5, 3.8, ..., 1.2],  # 位置 1 的预测
                                    [1.0, 0.0, 2.5, ..., -1.0],  # 位置 2 的预测
                                    ...
                                    [1.5, 2.8, -1.2, ..., 0.9]]  # 位置 6 的预测 (最后一个)
                    截取后 logits: [1.5, 2.8, -1.2, ..., 0.9] (位置 6 的预测，代表下一个 token 的分数)
            """
            logits = logits[:, -1, :] # 只取最后一个时间步

            # 温度采样
            """
            输入: logits [1, 65]
            输出: logits [1, 65] (经过温度调整)

            温度的作用:
                temperature     效果
                = 1.0           原始分布 (默认)
                > 1.0           分布更平坦 -> 更随机、更多样
                < 1.0           分布更尖锐 -> 更确定、更保守
                -> 0            退化为 argmax (总是选概率最高的)

            示例:
                原始logits: [2.0, 4.0, 1.0]

                temperature = 0.5:
                    logits / 0.5 = [4.0, 8.0, 2.0] 差距拉大，更确定
                    softmax后: [0.02, 0.96, 0.02] 第二个词 96% 的概率
                temperature = 2.0:
                    logits / 2.0 = [1.0, 2.0, 0.5] 差距缩小，更随机
                    softmax后: [0.28, 0.58, 0.14] 第二个词 58% 的概率
            """
            logits = logits / temperature

            # top-K 采样
            """
            输入: logits [1, 65]
            输出: logits [1, 65] (经过top-K 处理)

            作用: 只从概率最高的k个词中采样，避免低质量词

                示例: (top_K=3)
                原始 logits: [2.0, 4.0, 1.0, 3.5, 0.5, ...] (65个)
                Top-3值: [4.0, 3.5, 2.0] (对应词索引 1, 3, 0)
                掩码后: [2.0, 4.0, -inf, 3.5, -inf, ...] (只有索引 1, 3, 0 保留，其他词被置为 -inf)

                    Q: Top-K vs Top-P
                    A: Top-K 是从前K个词中采样，固定数量。Top-P 是从前P%的词中采样，按累积概率动态选择
            """
            if top_K is not None:
                v, _ = torch.topk(logits, min(top_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 转换为概率
            """
            输入: logits [1, 65]
            输出: probs [1, 65]

            物理意义：将分数归一化为概率，所有词概率之和 = 1

            数值示例：
                logits: [1.5, 2.8, -1.2, ..., 0.9]  (65个)
                probs: [0.05, 0.18, 0.01, ..., 0.03] (和为1.0)
            """
            probs = torch.softmax(logits, dim=-1)

            # 采样
            """
            输入: probs [1, 65]
            输出: idx_next [1, 1]

            物理意义: 按概率分布随机选择1个token
            示例:
                probs: [0.05, 0.18, 0.01, ..., 0.03] (65个)
                采样结果可能是:
                    - 索引 1 (概率 0.18)，对应字符 "R"
                    - 索引 0 (概率 0.05)，对应字符 "A"
                    - 索引 3 (概率 0.25)，对应字符 " " 最可能

                    Q: 为什么不用 argmax?
                    A: argmax总是选概率最高的，输出单一重复
                        multinomial 采样增加多样性，生成更自然
            """
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接
            """
            输入: start [1, 6], idx_next [1, 1]
            输出: start [1, 7]

            物理意义: 将新生成的token添加到序列
            数值示例:
                原始 start: [[45, 52, 49, 43, 52, 37]] (对应 "ROMEO:")
                idx_next: [[38]] (对应 " " 空格)
                拼接后 start: [[45, 52, 49, 43, 52, 37, 38]] (对应 "ROMEO: ")

            Q: 为什么 dim=1？
            A: dim=0是batch维度，dim=1是序列维度，我们要在序列维度上拼接新token
            """
            start = torch.cat((start, idx_next), dim=1)

    # 解码生成的文本
    """
    输入: start [1, 6 + max_new_tokens]
    输出: 字符串 "REMEO: "

    start[0].tolist() 将张量转换为列表: [45, 52, 49, 43, 52, 37, 38, ...] (对应生成的token ID序列)
    decode后: 根据tokenizer的映射关系，将ID序列转换回字符串 "ROMEO: " (包含原始提示和生成的新文本)
    """
    generate_text = tokenizer.decode(start[0].tolist())
    return generate_text


# 交互式对话
def interactive_chat(model, tokenizer) -> None:
    print("\n" + "=" * 60)
    print(" Tiny LLM 交互式对话 （输入 exit 退出）")
    print("=" * 60)

    while True:
        prompt = input("\n 👨 你")
        if prompt.lower() in ['exit', 'q']:
            print(" 再见 ヾ(•ω•`)o")
            break
        
        if not prompt.strip():
            continue

        # 生成回复
        print(" 🤖 AI:", end="", flush=True)
        response = generate_text(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_K=40
        )

        # 只显示生成的回复部分，去掉prompt
        generated = response[len(prompt):]
        print(generated)


# 批量生成示例
def batch_generate(model, tokenizer) -> None:
    print("\n" + "=" * 60)
    print(" Tiny LLM 批量生成示例")
    print("=" * 60)

    prompts = [
        "ROMEO:",
        "JULIET:",
        "KING RICHARD:",
        "To be or not to be,",
        "Once upon a time",
    ]

    for prompt in prompts:
        print(f"\n 👨 输入: {prompt}")
        response = generate_text(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_K=30,
            seed=42
        )

        generated = response[len(prompt):]
        print(f" 🤖 输出: {generated[:150]}...")


if __name__ == "__main__":
    import sys

    # 默认加载最佳模型
    model_path = "best_model.pth"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # 检查模型文件是否存在
    import os
    if not os.path.exists(model_path):
        print(f" 模型文件不存在: {model_path}")
        print(" 请先运行 train.py 训练模型，生成 best_model.pth")
        sys.exit(1)
    
    # 加载模型
    model, tokenizer = load_model(model_path)

    print("\n 选择模式:")
    print(" 1. 批量生成示例")
    print(" 2. 交互式对话")
    choice = input(" 请输入 1 或 2: ").strip()

    if choice == "1":
        batch_generate(model, tokenizer)
    elif choice == "2":
        interactive_chat(model, tokenizer)
    else:
        print(" 无效选择，默认运行批量生成")
        batch_generate(model, tokenizer)
