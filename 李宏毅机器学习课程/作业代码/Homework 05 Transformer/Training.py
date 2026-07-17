import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from collections import Counter

# 全局配置与特殊符号
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']

# 词表构建与分词 (Tokenizer & Vocab)
def basic_english_tokenizer(text):
    return text.lower().split()

def basic_chinese_tokenizer(text):
    return list(text)  # 中文按字切分

class SimpleVocab:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    def build_vocab(self, text_list):
        counter = Counter()
        for text in text_list:
            counter.update(self.tokenizer(text))
        for word, _ in counter.items():
            if word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def __len__(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi.get(tok, UNK_IDX) for tok in self.tokenizer(text)]


# 数据集与 DataLoader
class BilingualDataset(Dataset):
    def __init__(self, en_text, zh_text, min_ratio=0.8, max_ratio=5.0):
        self.en_text = []
        self.zh_text = []
        for en, zh in zip(en_text, zh_text):
            ratio = len(en) / len(zh)
            if min_ratio <= ratio <= max_ratio:
                self.en_text.append(en.strip())
                self.zh_text.append(zh.strip())

    def __len__(self):
        return len(self.en_text)

    def __getitem__(self, idx):
        return self.en_text[idx], self.zh_text[idx]


def get_collate_fn(vocab_en, vocab_zh, max_len=128):
    def collate_fn(batch):
        en_batch, zh_batch = [], []
        for en_text, zh_text in batch:
            # 编码并在头尾加上 <bos> 和 <eos>
            en_ids = [BOS_IDX] + vocab_en.encode(en_text) + [EOS_IDX]
            zh_ids = [BOS_IDX] + vocab_zh.encode(zh_text) + [EOS_IDX]

            # 强行砍掉超过最大长度的部分，保护显存不被占满
            en_ids = en_ids[:max_len]
            zh_ids = zh_ids[:max_len]

            en_batch.append(torch.tensor(en_ids, dtype=torch.long))
            zh_batch.append(torch.tensor(zh_ids, dtype=torch.long))

        # 补齐长度并堆叠成 [batch_size, seq_len]
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX,batch_first=True)
        zh_batch = pad_sequence(zh_batch, padding_value=PAD_IDX,batch_first=True)
        return en_batch, zh_batch
    return collate_fn

# 掩码生成 (Masking)
def generate_square_subsequent_mask(sz):
    # 生成对角线以上的三角形为 True，表示需要“遮掩”（不能偷看未来词）
    mask = torch.triu(torch.ones((sz, sz), device=DEVICE, dtype=torch.bool), diagonal=1)
    return mask


def create_mask(src, tgt):
    # 因为 batch_first=True，所以 src.shape[1] 才是句子长度！
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE, dtype=torch.bool)
    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



# 模型架构 (Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # 让形状从 [maxlen, emb_size] 变成 [1, maxlen, emb_size]
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        # 因为 batch_first=True，token_embedding 的形状是 [batch, seq_len, emb_size]
        # pos_embedding 也要对应取 seq_len 的维度
        return self.dropout(token_embedding + self.pos_embedding[:,:token_embedding.size(1), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size,
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)


# 训练循环 (Main & Train Loop)
if __name__ == "__main__":
    # 读取原始数据
    raw_en = []
    with open('raw_en.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # 使用 strip() 去除每行末尾自带的换行符 (\n)
            raw_en.append(line.strip())

    raw_zh = []
    with open('raw_zh.txt', 'r', encoding='utf-8') as file:
        for line in file:
            raw_zh.append(line.strip())

    # 初始化词表并构建
    vocab_en = SimpleVocab(basic_english_tokenizer)
    vocab_zh = SimpleVocab(basic_chinese_tokenizer)
    vocab_en.build_vocab(raw_en)
    vocab_zh.build_vocab(raw_zh)

    # 准备 Dataset 和 DataLoader
    dataset = BilingualDataset(raw_en, raw_zh)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=get_collate_fn(vocab_en, vocab_zh)
    )

    # 实例化模型
    model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=256,
        nhead=8,
        src_vocab_size=len(vocab_en),
        tgt_vocab_size=len(vocab_zh),
        dim_feedforward=512
    ).to(DEVICE)

    # 定义损失函数和优化器 (忽略 PAD 索引的损失计算)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 开始简单的训练循环 (以 5 个 Epoch 为例)
    print("开始训练...")
    model.train()

    for epoch in range(5):
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(DEVICE), tgt_batch.to(DEVICE)

            # 原来是 tgt_batch[:-1, :] 和 tgt_batch[1:, :]
            # 修改为：(逗号前面是 batch 维，全保留；逗号后面是 seq 维，做切片)
            tgt_input = tgt_batch[:, :-1]
            tgt_out = tgt_batch[:, 1:]

            # 生成掩码
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input)

            # 前向传播
            logits = model(
                src=src_batch,
                trg=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            # 计算损失：重塑 logits [seq_len, batch_size, vocab_size] -> [seq_len * batch_size, vocab_size]
            optimizer.zero_grad()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/5 - Loss: {total_loss / len(dataloader):.4f}")

    print("训练完成！模型已可以使用。")

    # 在训练全部结束后：
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_en_stoi': vocab_en.stoi,
        'vocab_en_itos': vocab_en.itos,
        'vocab_zh_stoi': vocab_zh.stoi,
        'vocab_zh_itos': vocab_zh.itos
    }

    torch.save(checkpoint, 'transformer_translation_model.pth')
    print("模型及词表已完整打包保存到 transformer_translation_model.pth")