# few sketches of the model implementation, needs sanity check
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_TOKEN = 2
EOS_TOKEN = 3
MAX_LENGTH = 30

# Sequence to Sequence Learning with Neural Networks
class EncoderRNN(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, n_layers=2, dropout_p=0.1):
    super(EncoderRNN, self).__init__()
    self.embedding = nn.Embedding(input_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True) # batch_first for (batch_size, seq_length)
    self.dropout = nn.Dropout(dropout_p)
  
  def forward(self, input): # (batch_size, seq_length)
    embedded = self.dropout(self.embedding(input)) # (batch_size, seq_length, embedding_size)
    _, (hidden, cell) = self.lstm(embedded) # output(batch_size, seq_length, hidden_size)
    return hidden, cell # feed this as initial (state, cell) to the decoder lstm

class DecoderRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, output_size, n_layers=2, dropout_p=0.1):
    super(DecoderRNN, self).__init__()
    self.embedding = nn.Embedding(output_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
    self.dropout = nn.Dropout(dropout_p)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, encoder_output, encoder_hidden, encoder_cell, target_tensor=None):
    batch_size = encoder_output.shape[0]
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_TOKEN) # batch of start token
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    decoder_outputs = []
    for i in range(MAX_LENGTH):
      decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, decoder_hidden, decoder_cell)
      decoder_outputs.append(decoder_output)
      if target_tensor is not None:
        decoder_input = target_tensor[:, i].unsqueeze(1) # teacher forcing
      else:
        decoder_input = decoder_output.argmax(dim=-1, keepdim=True).squeeze(-1)

    decoder_outputs = torch.cat(decoder_outputs, dim=1) # concat output token (batch_size, seq_length, output_size)
    decoder_outputs = F.log_softmax(decoder_outputs, dim=-1) # probs (batch_size, seq_length, output_size)

    return decoder_outputs

  def forward_step(self, input, hidden, cell):
    output = self.embedding(input)
    output = self.dropout(output)
    output, (hidden, cell) = self.lstm(output, (hidden, cell))
    output = self.fc(output)
    return output, hidden, cell


# Neural Machine Translation by Jointly Learning to Align and Translate
class EncoderBiRNN(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout_p=0.1):
    super(EncoderBiRNN, self).__init__()
    self.embedding = nn.Embedding(input_size, embed_size)
    self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_size * 2, hidden_size)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, input):
    embedded = self.dropout(self.embedding(input)) # (batch_size, seq_length, embed_size)
    outputs, hidden = self.gru(embedded) # output(batch_size, seq_length, hidden_size * 2) , hidden(n_layers * 2, batch_size, hidden_size)
    hidden_cat = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1) # concat forward and backward final hidden state(batch_size, hidden_size * 2)
    hidden_out = self.fc(hidden_cat) # initial decoder hidden state (batch_size, hidden_size)
    return outputs, hidden_out

class BahdanauAttention(nn.Module):
  def __init__(self, hidden_size):
    super(BahdanauAttention, self).__init__()
    self.wa = nn.Linear(hidden_size, hidden_size, bias=False)
    self.ua = nn.Linear(hidden_size * 2, hidden_size, bias=False) # encoder output is bidirectional
    self.va = nn.Linear(hidden_size, 1, bias=False)
  
  def forward(self, hidden, encoder_outputs):
    # hidden: (batch_size, hidden_size)
    # encoder_outputs: (batch_size, seq_length, hidden_size*2)
    hidden = hidden.unsqueeze(1) # (batch_size, 1, hidden_size)
    energy = torch.tanh(self.wa(hidden) + self.ua(encoder_outputs)) # (batch_size, seq_length, hidden_size)
    scores = self.va(energy) # (batch_size, seq_length, 1)
    scores = scores.squeeze(2) # (batch_size, seq_length)
    attn_weights = F.softmax(scores, dim=-1) # attention weights (batch_size, seq_length)
    context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # (batch_size, 1, hidden_size * 2)
    return context, attn_weights

class AttnDecoder(nn.Module):
  def __init__(self, output_size, embed_size, hidden_size, n_layers, dropout_p=0.5):
    super(AttnDecoder, self).__init__()
    self.attention = BahdanauAttention(hidden_size)
    self.embedding = nn.Embedding(output_size, embed_size)
    self.gru = nn.GRU(embed_size + hidden_size * 2, hidden_size, n_layers, batch_first=True, bidirectional=False) # input embedded + context vector
    self.fc = nn.Linear(embed_size + hidden_size + hidden_size * 2, output_size) # input embedded + gru output + context vector
    self.dropout = nn.Dropout(dropout_p)
  
  def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
    batch_size = encoder_outputs.shape[0]
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_TOKEN) # first start token (batch_size, 1)
    decoder_hidden = encoder_hidden
    decoder_outputs = []
    attentions = []
  
    for i in range(MAX_LENGTH):
      decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
      decoder_outputs.append(decoder_output)
      attentions.append(attn_weights)
      if target_tensor is not None:
        decoder_input = target_tensor[:, i].unsqueeze(1) # teacher forcing
      else:
        decoder_input = decoder_output.argmax(dim=-1, keepdim=True).squeeze(-1) # top 1
    
    decoder_outputs = torch.cat(decoder_outputs, dim=1) # (batch_size, seq_length, output_size)
    decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
    attentions = torch.cat(attentions, dim=1)
  
    return decoder_outputs
  
  def forward_step(self, input, hidden, encoder_outputs):
    # input: (batch_size, 1)
    # hidden: (batch_size, hidden_size)
    # encoder_outputs: (batch_size, seq_length, hidden_size * 2)
    embedded = self.dropout(self.embedding(input)) # (batch_size, 1, embed_size)
    context, attn_weights = self.attention(hidden, encoder_outputs)
    # context: (batch_size, 1, hidden_size * 2)
    # attn_weights: (batch_size, seq_length)
    input_gru = torch.cat([embedded, context], dim=2) # (batch_size, 1, embed_size + hidden_size * 2)
    output_gru, hidden = self.gru(input_gru, hidden.unsqueeze(0))
    # output_gru: (batch_size, 1, hidden_size)
    # hidden: (1, batch_size, hidden_size)
    hidden = hidden.squeeze(0) # (batch_size, hidden_size)
    output = self.fc(torch.cat([embedded, output_gru, context], dim=2)) # (batch_size, 1, output_size)
    return output, hidden, attn_weights