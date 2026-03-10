- Token
    Token 是大模型能理解的最小文本单位
    就像人类用"字"或"词"思考，大模型用 Token 思考
    Token 的三种粒度 (从粗到细)
        - 字符级 (Character-level)
            文本: "Hello"
            Token划分: ['H', 'e', 'l', 'l', 'o']
            Token数量: 5个
            优点: 
                1. 词表极小 (英文 ~65个字符, 中文 ~3000常用字)
                2. 不会出现OOV问题
                3. 实现简单, 适合学习
            缺点:
                1. 序列太长 ("人工智能" -> 6个Token), 计算慢
                2. 模型需要更深层才能理解"词"的含义

        - 词级 (Word-level)
            文本: "Hello world"
            Token 划分: ['Hello', 'world']
            Token 数量: 2 个
            优点：
                1. 序列段，计算快
                2. 每个Token有完整语义
            缺点：
                1. 词表巨大 (英文10万+， 中文50万+)
                2. 遇到新词 (如"LLaMA-3")无法处理 —> OOV问题
                3. 中文需要额外分词工具 (结巴等)
        
        - 子词级 (Subword) 工业界主流 (BPE/WordPiece)
            文本: "Hello LLaMA-3"
            Token 划分: ['Hello', 'LL', 'a', 'MA', '-3']
            Token 数量: 5个
            原理：
                1. 高频词保持完整: 'Hello' -> 1个Token
                2. 低频词拆分成字句: 'LLaMA-3' -> 多个Token
                3. 平衡词表大小和序列长度
            优点：
                1. 词表适中 (3 ~ 10万)
                2. 能处理新词 (拆分成已知子词)
                3. 序列长度合理
            代表算法:
                1. BPE (Byte Pair Encoding): GPT系列
                2. WordPiece: BERT, QWen
                3. SentencePiece: LLaMA, 支持多种语言
