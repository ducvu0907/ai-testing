import tfm
import torch

sentences = [
  "**The quick** brown fox *jumps* over **the lazy** dog!",
  "*A stitch* in **time** saves **nine, or so** they say...",
  "In a **world** full of *chaos,* **find your** calm!",
  "The stars **twinkled** like *diamonds* in the **night** sky!",
  "Every **cloud** has a *silver lining,* even on the *rainiest* days."
]
# print(preprocess(mock_sentences[0]))

d_model = 512
pos = 100
# print(sinusoid_positional_encoding(pos, d_model))

x = torch.tensor([[2, 5, 3, 7, 8, 10], [1, 2, 4, 5, 6, 7]])
positional_encoding = tfm.PositionalEncoding(d_model, 20)
# print(positional_encoding(x).shape)

x = torch.randn(2, 3, 512)
n_head = 8
multihead_attention = tfm.MultiHeadAttention(d_model, n_head)
# print(tfm.multihead_attention.split(x).shape)
x = torch.randn(2, 8, 3, 64)
# print(tfm.multihead_attention.concat(x).shape)

# src_data, tgt_data = tfm.load_data(tfm.src_path, tfm.tgt_path)
# print(len(src_data), len(tgt_data))

# src_tokens = tfm.tokenize(src_data)

# print(src_tokens)
