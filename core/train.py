"""
train.py - Nano LLM 训练循环

实现完整的训练流程，包括:
- 训练 / 验证 循环
- 学习率调度 (Warmup + Cosine Decay)
- 混合精度训练
- 模型保存与加载
- 训练进度可视化
    
    Q: 为什么需要学习预热 (Warmup)？
    Q: 梯度累积的作用是什么？
    Q: 如何防止过拟合？
"""

import os
import time
import math
import torch
from torch.nn import functional as F

# 导入配置和模块
from config.config import (
    device, block_size, batch_size, train_split,
    n_embd, n_head, n_layer, dropout,
    learning_rate, max_iters, eval_interval, eval_iters, warmup_iters
)
from core.data import prepare_data, SimpleTokenizer, TextDataset
from core.model import GPT


# 学习率调度器 (Learning Rate Scheduler)
def get_lr(it: int) -> float:
    """
    学习率调度: Warmup + Cosine Decay

        Q: 为什么需要Warmup?
        A: 训练初期梯度不稳定，小学习率避免模型“跑偏”

        Q: 为什么要用 Cosine Decay?
        A: 平滑降低学习率，帮助模型收敛到更优解 
    """

    # 预热阶段: 线性增加学习率
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 衰减阶段: 余弦退火
    if it > max_iters:
        return learning_rate * 0.1
    
    # 余弦衰减
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff


# 训练主函数
def train():

    # 数据准备
    print("Preparing data...")
    text = prepare_data()
    tokenizer = SimpleTokenizer(text)
    vocab_size = tokenizer.vocab_size

    train_dataset = TextDataset(text, tokenizer, block_size, split="train")
    val_dataset = TextDataset(text, tokenizer, block_size, split="val")

    # 初始化模型
    print("初始化模型...")
    model = GPT(vocab_size)
    model.to(device)

    # 初始化优化器
    """
        Q: 为什么用 AdamW 而不是 Adam?
        A: AdamW 修正了权重衰减的实现，对Transformer更稳定
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),    # Transformer 常用配置
        weight_decay=1e-1,    # 正则化，防止过拟合
    )

    # 训练循环
    print("开始训练...")
    start_time = time.time()

    # 用于记录最佳验证loss
    best_val_loss = float('inf')

    for iter in range(max_iters):
        # 学习率调度
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 获取训练批次
        X, Y = train_dataset.get_batch(batch_size)

        # 前向传播
        """
            Q: 为什么训练时要传 targets?
            A: 需要计算 Loss 用于反向传播
        """
        logits, loss = model(X, Y)

        # 反向传播
        # 清空之前的梯度 (PyTorch 默认累积梯度)
        optimizer.zero_grad(set_to_none=True)

        # 计算梯度
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        """
            Q: 为什么需要梯度裁剪?
            A: Transformer 容易出现梯度爆炸，裁剪保证训练稳定
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新权重
        optimizer.step()

        # 进度打印与评估
        if (iter + 1) % eval_interval == 0 or iter == max_iters - 1:
            
            # 计算验证集 Loss
            val_loss = estimate_loss(model, val_dataset, eval_iters)

            # 计算耗时
            elapsed_time = time.time() - start_time
            iter_per_sec = (iter + 1) / elapsed_time if elapsed_time > 0 else 0

            # 打印进度
            print(f"📍 Step {iter + 1}/{max_iters}")
            print(f"   Train Loss: {loss.item():.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   LR:         {lr:.2e}")
            print(f"   Speed:      {iter_per_sec:.2f} iters/sec")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, tokenizer, f"best_model.pth")
                print(f" 📂 保存最佳模型 (val_loss: {val_loss:.4f})")

            # 生成样本测试
            model.eval()
            with torch.no_grad():
                # 用莎士比亚角色名作为提示
                prompts = ["ROMEO:", "JULIET:", "KING:"]
                for prompt in prompts:
                    start = torch.tensor(
                        [[tokenizer.string_to_int[c] for c in prompt]],
                        device=device
                    )
                    generated = model.generate(start, max_new_tokens=100)
                    text = tokenizer.decode(generated[0].tolist())
                    print(f"   🔮 {text[:90]}...")
            model.train()

            print("-" * 60)


    # 训练完成
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成！")
    print(f"   总耗时：{total_time / 60:.2f} 分钟")
    print(f"   最佳验证 Loss: {best_val_loss:.4f}")

    # 保存最终模型
    save_model(model, tokenizer, f"final_model.pth")
    print(f" 📂 保存最终模型 (final_model.pth)")


# 验证集评估函数
@torch.no_grad()
def estimate_loss(model, dataset, eval_iters: int) -> float:
    """
    在验证集上评估模型

        Q: 为什么用 torch.no_grad()?
        A: 禁用梯度计算，节省显存和计算时间
    """
    model.eval()
    losses = torch.zeros(eval_iters, device=device)

    for k in range(eval_iters):
        X, Y = dataset.get_batch(batch_size)
        logits, loss = model(X, Y)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


# 模型保存函数
def save_model(model, tokenizer, filepath: str):
    """
    保存模型权重和分词器

        Q: 为什么保存 state_dict 而不是整个模型?
        A: state_dict 只保存参数，文件更小，加载更灵活
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'stoi': tokenizer.string_to_int,
        'itos': tokenizer.int_to_string,
        'config': {
            'block_size': block_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f" 模型已保存至: {filepath}")
    

# 模型加载函数 (推理使用)
def load_model(filepath: str):
    """
    加载保存的模型

    Returns:
        model: GPT模型
        tokenizer: SimpleTokenizer
    """
    checkpoint = torch.load(filepath, map_location=device)

    # 重建分词器
    tokenizer = SimpleTokenizer("")
    tokenizer.string_to_int = checkpoint['stoi']
    tokenizer.int_to_string = checkpoint['itos']
    tokenizer.vocab_size = checkpoint['vocab_size']

    # 重建模型
    model = GPT(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f" 模型已从 {filepath} 加载...")
    return model, tokenizer


if __name__ == "__main__":
    train()