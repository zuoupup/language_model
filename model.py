import torch
import torch.nn as nn
import math 
import re
from collections import Counter

VOCAB_SIZE = 10000

# Load custom dataset
print("Loading custom dataset...")
with open("custom_dataset.txt", "r", encoding="utf-8") as f:
    sentences = f.readlines()

# Updated tokenizer
def tokenizer(x):
    return re.findall(r"\w+|[^\s\w]", x.strip())

# Build vocabulary
all_words = [word for sentence in sentences for word in tokenizer(sentence)]
counter = Counter(all_words)
most_common = counter.most_common(VOCAB_SIZE - 2)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
idx_to_word = {idx: word for word, idx in vocab.items()}
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# Transformer Language Model with causal mask
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(embedding_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def forward(self, src, src_key_padding_mask):
        # src shape: (batch_size, seq_len)
        src_emb = self.embedding(src)  # (batch_size, seq_len, embedding_dim)
        src_emb = self.pos_encoder(src_emb)

        # Create causal mask
        seq_len = src_emb.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(src_emb.device)

        output = self.transformer_encoder(src_emb, mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)  # (batch_size, seq_len, vocab_size)
        return output