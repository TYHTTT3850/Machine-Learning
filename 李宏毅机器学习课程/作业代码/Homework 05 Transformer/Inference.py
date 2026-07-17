import torch
import torch.nn as nn
import math

# 全局配置与基础类 (和训练时一致)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def basic_english_tokenizer(text):
    return text.lower().split()


def basic_chinese_tokenizer(text):
    return list(text)

class SimpleVocab:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi.get(tok, UNK_IDX) for tok in self.tokenizer(text)]


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz), device=DEVICE, dtype=torch.bool), diagonal=1)



# 模型架构定义 (和训练时一致)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


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

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)



# 推理
def translate_sentence(model, sentence, vocab_en, vocab_zh, device, max_len=50):
    model.eval()
    tokens = [BOS_IDX] + vocab_en.encode(sentence) + [EOS_IDX]
    src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device, dtype=torch.bool)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        tgt_indexes = [BOS_IDX]

        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_indexes, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.shape[1])
            out = model.decode(tgt_tensor, memory, tgt_mask)
            prob = model.generator(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            next_word_idx = next_word.item()

            tgt_indexes.append(next_word_idx)
            if next_word_idx == EOS_IDX:
                break

    translated_words = [vocab_zh.itos.get(idx, "<unk>") for idx in tgt_indexes]
    return "".join(translated_words[1:-1])

# 交互
if __name__ == "__main__":
    print("正在加载模型和词表...")

    # 读取保存的打包文件 (map_location 保证如果在没有显卡的电脑上也能强行用 CPU 跑)
    checkpoint = torch.load('transformer_translation_model.pth', map_location=DEVICE)

    # 恢复中英文词表
    vocab_en = SimpleVocab(basic_english_tokenizer)
    vocab_zh = SimpleVocab(basic_chinese_tokenizer)
    vocab_en.stoi = checkpoint['vocab_en_stoi']
    vocab_en.itos = checkpoint['vocab_en_itos']
    vocab_zh.stoi = checkpoint['vocab_zh_stoi']
    vocab_zh.itos = checkpoint['vocab_zh_itos']

    # 实例化空壳模型 (参数和训练时设定的大小一致)
    model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=256,
        nhead=8,
        src_vocab_size=len(vocab_en),
        tgt_vocab_size=len(vocab_zh)
    ).to(DEVICE)

    # 4. 将保存的权重注入模型
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 开启评估模式

    print("\n" + "=" * 40)
    print("加载成功！进入翻译模式 (输入 'q' 退出)")
    print("=" * 40)

    # 5. 开启测试循环
    while True:
        text = input("\n请输入英文句子: ")
        if text.strip().lower() == 'q':
            break
        if not text.strip():
            continue

        result = translate_sentence(model, text, vocab_en, vocab_zh, DEVICE)
        print(f"翻译结果: {result}")