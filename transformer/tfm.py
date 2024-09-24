import torch
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.utils
import torch.utils.data
import tqdm
import re
import math

device = torch.device("cuda" if torch.cpu.is_available() else "cpu")
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"
unk_token = "<unk>"
src_path = "data/train.en.txt"
tgt_path = "data/train.vi.txt"

# data 
def preprocess(sentence):
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
  sentence = re.sub(r"[^\w\s?.!,¿]+", "", sentence)
  sentence = ' '.join(sentence.split())
  return sentence

def tokenize(text):
  return [[sos_token] + word_tokenize(preprocess(snt)) + [eos_token] for snt in text]

def build_vocab(tokens_list, special_tokens):
  vocab = {}
  for token in special_tokens:
    vocab[token] = len(vocab)

  tokens_set = set([token for tokens in tokens_list for token in tokens])
  for token in tokens_set:
    if token not in vocab:
      vocab[token] = len(vocab)

  return vocab

def load_data(src_path, tgt_path):
  src_data = open(src_path, 'r').read().split('\n')
  tgt_data = open(tgt_path, 'r').read().split('\n')
  # print(len(src_data), len(tgt_data))
  # remove empty lines
  src = [line for i, line in enumerate(src_data) if line and tgt_data[i]]
  tgt = [line for i, line in enumerate(tgt_data) if line and src_data[i]]
  return src, tgt

def get_collate_fn(pad_index):
  def collate_fn(data):
    src, tgt = zip(*data)
    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_index)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_index)
    return src_padded, tgt_padded
  return collate_fn

def get_data_loader(src_path, tgt_path, src_vocab, tgt_vocab, batch_size, pad_index, shuffle):
  src_data, tgt_data = load_data(src_path, tgt_path)
  dataset = NMTDataset(src_data, tgt_data, src_vocab, tgt_vocab, pad_index)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle, collate_fn=get_collate_fn(pad_index))
  return data_loader

class NMTDataset(torch.utils.data.Dataset):
  def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
    self.src_data = torch.tensor(src_data, dtype=torch.long)
    self.tgt_data = torch.tensor(tgt_data, dtype=torch.long)
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab

  def __len__(self):
    assert len(self.src_data) == len(self.tgt_data), "source and target files must have same number of lines"
    return len(self.src_data)

  def __getitem__(self, idx):
    src = self.src_data[idx]
    tgt = self.tgt_data[idx]
    return src, tgt

# model
class PositionalEncoding(torch.nn.Module):
  def __init__(self, d_model, max_len):
    # sinusoid
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = torch.zeros((max_len, d_model)) # (max_len, d_model)
    self.pos_encoding.requires_grad = False # no gradient
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
    _2i = torch.arange(0, d_model, step=2, dtype=torch.long) # (d_model)
    # pe(pos, i) = sin(pos) / 10000^(2i / d_model) or cos(pos) / 10000^(2i / d_model)
    angles = pos / (10000 ** (_2i / d_model))
    self.pos_encoding[:, 0::2] = torch.sin(angles)
    self.pos_encoding[:, 1::2] = torch.cos(angles)

  def forward(self, x):
    # x: (batch_size, seq_len)
    _, seq_len = x.shape
    return self.pos_encoding[:seq_len, :].unsqueeze(0) # (1, max_len, d_model)

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_model, n_head):
    super(MultiHeadAttention, self).__init__()
    self.n_head = n_head
    self.w_q = torch.nn.Linear(d_model, d_model)
    self.w_k = torch.nn.Linear(d_model, d_model)
    self.w_v = torch.nn.Linear(d_model, d_model)
    self.w_o = torch.nn.Linear(d_model, d_model)

  def scale_dot_product_attention(self, q, k, v, mask=None):
    # q: (batch_size, n_head, q_length, d_q)
    # k: (batch_size, n_head, k_length, d_k)
    # v: (batch_size, n_head, v_length, d_v)
    d_k = k.shape[-1]
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k) # (batch_size, n_head, q_length, k_length)
    if mask:
      scores = scores.masked_fill(mask == 0, float("-1e12")) # masked attention
    attn_weights = torch.nn.functional.softmax(scores, dim=-1) # (batch_size, n_head, q_length, k_length)
    out = torch.matmul(attn_weights, v) # (batch_size, n_head, q_length, d_v)
    return out, attn_weights

  # split into n heads
  def split(self, x):
    # x: (batch_size, seq_length, d_model)
    batch_size, seq_length, d_model = x.shape
    d_k = d_model // self.n_head
    x = x.view(batch_size, seq_length, self.n_head, d_k).transpose(1, 2)
    return x # (batch_size, n_head, seq_length, d_k)

  # inverse heads splitting
  def concat(self, x):
    # x: (batch_size, n_head, seq_length, d_K)
    batch_size, n_head, seq_length, d_k = x.shape
    d_model = d_k * n_head
    x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
    return x # (batch_size, seq_length, d_model)

  def forward(self, q, k, v, mask=None):
    q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
    q, k, v = self.split(q), self.split(k), self.split(v)
    out, _ = self.scale_dot_product_attention(q, k, v, mask)
    out = self.concat(out)
    out = self.w_o(out)
    return out # (batch_size, seq_length, d_model)

class PositionWiseFeedForward(torch.nn.Module):
  def __init__(self, d_model, d_ff, dropout_rate):
    super(PositionWiseFeedForward, self).__init__()
    self.fc1 = torch.nn.Linear(d_model, d_ff)
    self.fc2 = torch.nn.Linear(d_ff, d_model)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x

class EncoderLayer(torch.nn.Module):
  def __init__(self, n_head, d_model, d_ff, dropout_rate):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, n_head)
    self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.dropout1 = torch.nn.Dropout(dropout_rate)
    self.dropout2 = torch.nn.Dropout(dropout_rate)

  # x is input after word embedding + positional encoding
  # src is already padded so need to mask pad index to remove it from self attention scores
  def forward(self, x, src_mask):
    # multi headed attention
    _x = x
    x = self.self_attn(q=x, k=x, v=x, mask=src_mask)
    x = self.dropout1(x)
    x = self.norm1(x + _x) # residual + layer norm
    # feed forward
    _x = x
    x = self.ffn(x)
    x = self.dropout2(x)
    x = self.norm2(x + _x) # residual + layer norm
    return x # (batch_size, seq_length, d_model)

class Encoder(torch.nn.Module):
  def __init__(self, n_layer, n_head, vocab_size, d_model, d_ff, src_max_len, dropout_rate):
    super(Encoder, self).__init__()
    self.embedding = torch.nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, src_max_len)
    self.layers = torch.nn.ModuleList([EncoderLayer(n_head, d_model, d_ff, dropout_rate) for _ in range(n_layer)])

  def forward(self, x, src_mask):
    # x: (batch_size, seq_length)
    x = self.embedding(x) + self.pos_encoding(x) # (batch_size, seq_length, d_model)
    for layer in self.layers:
      x = layer(x, src_mask)
    return x

class DecoderLayer(torch.nn.Module):
  def __init__(self, n_head, d_model, d_ff, dropout_rate):
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, n_head)
    self.cross_attn = MultiHeadAttention(d_model, n_head)
    self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.norm3 = torch.nn.LayerNorm(d_model)
    self.dropout1 = torch.nn.Dropout(dropout_rate)
    self.dropout2 = torch.nn.Dropout(dropout_rate)
    self.dropout3 = torch.nn.Dropout(dropout_rate)

  def forward(self, x, enc_out, src_mask, tgt_mask):
    # masked multi head attention
    _x = x
    x = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
    x = self.dropout1(x)
    x = self.norm1(x + _x) # residual + layer norm
    # encoder multi head attention
    _x = x
    x = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
    x = self.dropout2(x)
    x = self.norm2(x + _x) # residual + layer norm
    # feed forward
    _x = x
    x = self.ffn(x)
    x = self.dropout3(x)
    x = self.norm3(x + _x) # residual + layer norm
    return x

class Decoder(torch.nn.Module):
  def __init__(self, n_layer, n_head, vocab_size, d_model, d_ff, tgt_max_len, dropout_rate):
    super(Decoder, self).__init__()
    self.embedding = torch.nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, tgt_max_len)
    self.layers = torch.nn.ModuleList([DecoderLayer(n_head, d_model, d_ff, dropout_rate) for _ in range(n_layer)])
    self.fc = torch.nn.Linear(d_model, vocab_size)
    self.softmax = torch.nn.Softmax(dim=-1)

  # training: target sentence + self masked attention + cross attention
  # inference: start with sos token, generate token by token, dynamic masked attention, cross attention until eos or max length
  def forward(self, x, enc_out, src_mask, tgt_mask):
    # x: (batch_size, seq_length)
    x = self.embedding(x) + self.pos_encoding(x) # (batch_size, seq_length, d_model)
    for layer in self.layers:
      x = layer(x, enc_out, src_mask, tgt_mask)
    out = self.fc(x) # (batch_size, seq_length, vocab_size)
    # out = self.softmax(x) # dont need cuz torch cross entropy loss 
    return out

class Transformer(torch.nn.Module):
  def __init__(self, enc_vocab_size, dec_vocab_size, d_model, n_layer, n_head, src_max_len, tgt_max_len, d_ff, dropout_rate, pad_index):
    super(Transformer, self).__init__()
    self.pad_index = pad_index
    self.encoder = Encoder(n_layer, n_head, enc_vocab_size, d_model, d_ff, src_max_len, dropout_rate)
    self.decoder = Decoder(n_layer, n_head, dec_vocab_size, d_model, d_ff, tgt_max_len, dropout_rate)

  def _get_src_mask(self, src):
    # src: (batch_size, src_length)
    src_mask = (src == self.pad_index).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, src_len)
    # broadcast -> (batch_size, n_head, src_len, src_len)
    return src_mask

  def _get_tgt_mask(self, tgt):
    # tgt: (batch_size, tgt_length)
    tgt_len = tgt.shape[1]
    tgt_pad_mask = (tgt == self.pad_index).unsqueeze(1).unsqueeze(3) # (batch_size, 1, tgt_len, 1)
    tgt_future_mask = torch.tril(torch.ones(tgt_len, tgt_len)) # (tgt_len, tgt_len)
    return tgt_pad_mask & tgt_future_mask # (batch_size, 1, tgt_len, tgt_len), broadcast -> (batch_size, n_head, tgt_len, tgt_len)

  def forward(self, src, tgt):
    # src, tgt: (batch_size, seq_length)
    src_mask = self._get_src_mask(src)
    tgt_mask = self._get_tgt_mask(tgt)
    enc_out = self.encoder(src, src_mask)
    dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
    return dec_out # (batch_size, tgt_length, vocab_size)

# train, evaluation
def train_fn(model, dataloader, optimizer, criterion, clip):
  model.train()
  epoch_loss = 0
  for i, (src, tgt) in tqdm(dataloader):
    src, tgt = src.to(device), tgt.to(device)
    # src, tgt: (batch_size, seq_length)
    optimizer.zero_grad()
    out = model(src, tgt[:, :-1]) # (batch_size, tgt_length - 1, vocab_size), dont take the last token
    out = out.contiguous().view(-1, out.shape[-1]) # (batch_size * (tgt_length - 1), vocab_size)
    tgt = tgt[:, 1:].contiguous().view(-1) # (batch_size * tgt_length - 1)
    loss = criterion(out, tgt)
    loss.backward()
    epoch_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    print(f"iter: {i}, loss: {loss.item():.4f}")
  return epoch_loss / len(dataloader)

def eval_fn(model, dataloader, criterion):
  pass

def greedy_search(model, src, max_length=50):
  pass
