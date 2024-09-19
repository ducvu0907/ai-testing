import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
import spacy
from torch.utils.data import Dataset, DataLoader
import tqdm
import math
from heapq import heappush, heappop

device = torch.device("cuda" if torch.cpu.is_available() else "cpu")
dataset = datasets.load_dataset("bentrevett/multi30k")

en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"
unk_token = "<unk>"

train_data, valid_data, test_data = (
  dataset["train"],
  dataset["validation"],
  dataset["test"],
)

def tokenize_data(data, en_nlp, de_nlp, sos_token, eos_token):
  en_tokens = [token.text.lower() for token in en_nlp.tokenizer(data["en"])]
  de_tokens = [token.text.lower() for token in de_nlp.tokenizer(data["de"])]
  en_tokens = [sos_token] + en_tokens + [eos_token]
  de_tokens = [sos_token] + de_tokens + [eos_token]
  return {"en_tokens": en_tokens, "de_tokens": de_tokens}

fn_kwargs = {
  "en_nlp": en_nlp,
  "de_nlp": de_nlp,
  "sos_token": sos_token,
  "eos_token": eos_token,
}

train_data = train_data.map(tokenize_data, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_data, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_data, fn_kwargs=fn_kwargs)

def build_vocab(token_lists, min_freq=1):
  s = set([token for tokens in token_lists for token in tokens])
  vocab = {}
  vocab[unk_token] = len(vocab)
  vocab[pad_token] = len(vocab)
  vocab[sos_token] = len(vocab)
  vocab[eos_token] = len(vocab)
  for token in s:
    if token not in vocab:
      vocab[token] = len(vocab)
  return vocab

en_vocab = build_vocab(train_data["en_tokens"])
de_vocab = build_vocab(train_data["de_tokens"])

pad_index = en_vocab[pad_token]
unk_index = en_vocab[unk_token]
sos_index = en_vocab[sos_token]
eos_index = en_vocab[eos_token]

assert en_vocab[pad_token] == de_vocab[pad_token]
assert en_vocab[unk_token] == de_vocab[unk_token]
assert en_vocab[sos_token] == de_vocab[sos_token]
assert en_vocab[eos_token] == de_vocab[eos_token]

def text_to_ids(texts, vocab):
  return [[vocab.get(token, unk_index) for token in text] for text in texts]

en_ids = text_to_ids(train_data["en_tokens"], en_vocab)
de_ids = text_to_ids(train_data["de_tokens"], de_vocab)

class TranslationDataset(Dataset):
  def __init__(self, src, tgt, src_vocab, tgt_vocab):
    self.src = src
    self.tgt = tgt
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab
    self.src_max_length = max(len(seq) for seq in src)
    self.tgt_max_length = max(len(seq) for seq in tgt)
  
  def __len__(self):
    return len(self.src)

  def __getitem__(self, idx):
    src_ids = self.src[idx]
    tgt_ids = self.tgt[idx]
    # padding sequence
    src_ids = src_ids + [pad_index] * (self.src_max_length - len(src_ids))
    tgt_ids = tgt_ids + [pad_index] * (self.tgt_max_length - len(tgt_ids))
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

    return src_tensor, tgt_tensor

def get_data_loader(data, batch_size, shuffle):
  data = data.map(tokenize_data, fn_kwargs=fn_kwargs)
  en_ids = text_to_ids(data["en_tokens"], en_vocab)
  de_ids = text_to_ids(data["de_tokens"], de_vocab)
  dataset = TranslationDataset(en_ids, de_ids, en_vocab, de_vocab)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return data_loader

# model architecture
class Encoder(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout_rate):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_size, embed_size)
    self.bigru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True, dropout=dropout_rate)
    self.fc = nn.Linear(hidden_size*2, hidden_size)
    self.dropout = nn.Dropout(dropout_rate)
  def forward(self, input):
    # input: (seq_length, batch_size)
    embedded = self.dropout(self.embedding(input)) # (seq_length, batch_size, embed_size)
    outputs, hidden = self.bigru(embedded)
    # output: (seq_length, batch_size, hidden_size*2)
    # hidden: (n_layers * n_directions, batch_size, hidden_size)
    hidden_cat = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
    # hidden_cat: (batch_size, hidden_size*2) - n_layers set to 1 here for simplicity
    hidden_out = F.relu(self.fc(hidden_cat))
    # hidden_out: (batch_size, hidden_size)
    return outputs, hidden_out

class BahdanauAttention(nn.Module):
  def __init__(self, hidden_size):
    super(BahdanauAttention, self).__init__()
    self.wa = nn.Linear(hidden_size, hidden_size, bias=False)
    self.ua = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    self.va = nn.Linear(hidden_size, 1, bias=False)
  def forward(self, hidden, encoder_outputs):
    # hidden: (batch_size, hidden_size)
    # encoder_outputs: (seq_length, batch_size, hidden_size*2)
    seq_length = encoder_outputs.shape[0]
    hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_size)
    encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, seq_length, hidden_size * 2)
    # calculate alignment scores
    energy = F.relu(self.wa(hidden) + self.ua(encoder_outputs))  # (batch_size, seq_length, hidden_size)
    scores = self.va(energy)  # (batch_size, seq_length, 1)
    scores = scores.squeeze(2)  # (batch_size, seq_length)
    
    # calculate the attention weights
    attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_length)
    
    # calculate context vector
    context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size * 2)
    
    # context_vector: (batch_size, 1, hidden_size * 2)
    # attn_weights: (batch_size, seq_length)
    return context_vector, attn_weights

class Decoder(nn.Module):
  def __init__(self, output_size, embed_size, hidden_size, n_layers, dropout_rate):
    super(Decoder, self).__init__()
    self.output_size = output_size
    self.embedding = nn.Embedding(output_size, embed_size)
    self.gru = nn.GRU(embed_size + hidden_size*2, hidden_size, n_layers, bidirectional=False, dropout=dropout_rate)
    self.attention = BahdanauAttention(hidden_size)
    self.fc = nn.Linear(embed_size + hidden_size + hidden_size*2, output_size)
    self.dropout = nn.Dropout(dropout_rate)
  # decoder takes 1 token at a time
  def forward(self, input, hidden, encoder_outputs):
    # input: (batch_size,)
    # hidden: (batch_size, hidden_size)
    # encoder_outputs: (seq_length, batch_size, hidden_size*2)
    embedded = self.dropout(self.embedding(input.unsqueeze(0))) # (1, batch_size, embed_size)
    context_vector, attn_weights = self.attention(hidden, encoder_outputs)
    # context_vector: (batch_size, 1, hidden_size*2)
    # attn_weights: (batch_size, seq_length)
    context_vector = context_vector.permute(1, 0, 2) # (1, batch_size, hidden_size*2)
    gru_input = torch.cat([embedded, context_vector], dim=2) # (1, batch_size, embed_size + hidden_size*2)
    output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
    # output: (1, batch_size, hidden_size)
    # hidden: (1, batch_size, hidden_size)
    embedded = embedded.squeeze(0) # (batch_size, embed_size)
    output = output.squeeze(0) # (batch_size, hidden_size)
    context_vector = context_vector.squeeze(0) # (batch_size, hidden_size*2)
    pred = self.fc(torch.cat([embedded, output, context_vector], dim=1)) # (batch_size, output_size)
    hidden = hidden.squeeze(0) # (batch_size, hidden_size)
    return pred, hidden, attn_weights

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio=0.5):
    # source: (source_seq_length, batch_size)
    # target: (target_seq_length, batch_size)
    target_seq_length, batch_size = target.shape
    target_vocab_size = self.decoder.output_size
    outputs = torch.zeros((target_seq_length, batch_size, target_vocab_size)).to(device)
    encoder_outputs, hidden = self.encoder(source)
    # encoder_outputs: (source_seq_length, batch_size, hidden_size*2)
    # hidden: (batch_size, hidden_size)
    input = target[0, :] # (batch_size,) - initial input is sos token
    for t in range(1, target_seq_length):
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
      # output: (batch_size, output_size)
      # hidden: (batch_size, hidden_size)
      outputs[t] = output
      top1 = output.argmax(dim=1)
      input = target[t] if np.random.random() < teacher_force_ratio else top1
    return outputs

def train_fn(model, dataloader, optimizer, criterion, clip=4.0):
  model.train()
  total_loss = 0

  for idx, (src, tgt) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), position=0, leave=True)):
    src, tgt = src.to(device), tgt.to(device)
    src, tgt = src.T, tgt.T
    # src: (src_seq_length, batch_size)
    # tgt: (tgt_seq_length, batch_size)
    optimizer.zero_grad()
    output = model(src, tgt) # (target_seq_length, batch_size, output_size)
    output = output.view(-1, output.shape[-1])  # (target_seq_length*batch_size, output_size)
    tgt = tgt.contiguous().view(-1)  # (target_seq_length*batch_size)
    loss = criterion(output, tgt)
    total_loss += loss.item()
    perplexity = math.exp(loss.item())
    loss.backward()
    output = output.argmax(dim=-1)
    nn.utils.clip_grad_norm_(model.parameters(), clip) # gradient clipping
    optimizer.step()
    if (idx + 1) % 50 == 0:
      print(f"loss: {loss.item()}, perplexity: {perplexity}")

  return total_loss / len(dataloader)
  
# train
INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(de_vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 1
DROPOUT = 0.5

encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder).to(device)

train_loader = get_data_loader(train_data, batch_size=64, shuffle=True)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab[pad_token]) # ignore padding token

# test train
num_epochs = 10
for epoch in range(num_epochs):
  epoch_loss = train_fn(model, train_loader, optimizer, criterion)
  print(f"epoch {epoch+1}, loss: {epoch_loss}")

def eval_fn(model, dataloader, criterion):
  model.eval()
  total_loss = 0
  total_correct = 0
  total_tokens = 0

  with torch.no_grad():
    for src, tgt in tqdm(dataloader, total=len(dataloader), position=0, leave=True):
      src, tgt = src.to(device), tgt.to(device)
      src = src.T  # (source_seq_length, batch_size)
      tgt = tgt.T  # (target_seq_length, batch_size)

      output = model(src, tgt, teacher_force_ratio=0)  # turn off teacher force during evaluation
      
      output = output.view(-1, output.shape[-1])  # (target_seq_length * batch_size, output_size)
      tgt = tgt.contiguous().view(-1)  # (target_seq_length * batch_size)
      loss = criterion(output, tgt)
      
      total_loss += loss.item()
      
      pred = output.argmax(dim=-1)  # (target_seq_length * batch_size)
      correct = pred.eq(tgt)  # (target_seq_length * batch_size)
      total_correct += correct.sum().item()
      total_tokens += tgt.size(0)

  loss = total_loss / len(dataloader)
  accuracy = total_correct / total_tokens

  print(f"evaluation loss: {loss:.4f}")
  print(f"evaluation accuracy: {accuracy:.4f}")

  return loss, accuracy

# map idx back to token
en_itos = {idx:token for token, idx in en_vocab.items()}
de_itos = {idx:token for token, idx in de_vocab.items()}

# greedy search translation
@torch.no_grad()
def greedy_decode(model, sentence, max_length=50):
  model.eval()
  tokens = [token.text.lower() for token in en_nlp.tokenizer(sentence)]
  token_ids = [sos_index] + [en_vocab[token] for token in tokens] + [eos_index]
  source = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(device) # (seq_length, 1)

  encoder_outputs, hidden = model.encoder(source)

  target_ids = [sos_index]
  attentions = [] # containing attention score of each target token to all source tokens

  for _ in range(max_length):
    input = torch.tensor([target_ids[-1]], dtype=torch.long)
    output, hidden, attention = model.decoder(input, hidden, encoder_outputs) 
    # output: (1, output_size)
    # hidden: (1, hidden_size)
    # attention: (1, source_seq_length)
    output_id = output.argmax(dim=-1)
    target_ids.append(output_id.item())
    attentions.append(attention.squeeze(0).item())

    if output_id == de_vocab[eos_token]:
      break
  
  # attentions: (target_seq_length, source_seq_length)
  target_tokens = [de_itos[id] for id in target_ids]
  return target_tokens, attentions

def visualize_attention_scores(source, target, attentions):
  plt.figure(figsize=(16, 10))
  sns.heatmap(attentions, xticklabels=source.lower(), yticklabels=target, cmap="viridis")
  plt.xlabel("input text")
  plt.ylabel("output text")
  plt.show()

# beam search translation
@torch.no_grad()
def beam_decode(model, sentence, max_length=50, beam_width=5):
  model.eval()
  tokens = [token.text.lower() for token in sentence]
  token_ids = [sos_index] + [en_vocab[token] for token in tokens] + [eos_index]
  source = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(device) # (seq_length, 1)

  encoder_outputs, hidden = model.encoder(source)
  queue = [(0, [sos_index]),] # sequence, log probability
  result_sequences = []

  for _ in range(max_length):
    if not queue:
      break
    score, curr_seq = heappop(queue)
    input = torch.tensor(curr_seq[-1], dtype=torch.long).to(device) # (1,)
    output, hidden, _ = model.decoder(input, hidden, encoder_outputs)
    # output: (1, output_size)
    # hidden: (1, hidden_size)
    # attention: (1, source_seq_length)
    topk = torch.topk(output, beam_width, dim=-1) # topk

    for i in range(len(topk)):
      new_score = score - topk.values[0][i].item() # log prob, minus because of max-heap
      new_seq = curr_seq + [topk.indices[0][i].item()]

      if new_seq[-1] == eos_index:
        result_sequences.append((-new_score, new_seq))
      else:
        heappush(queue, (new_score, new_seq))

    # remove branches
    while len(queue) > beam_width:
      heappop(queue)
      
  return max(result_sequences, key=lambda x: x[0])[1] if result_sequences else heappop(queue)[1]