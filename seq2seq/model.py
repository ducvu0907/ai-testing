import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_TOKEN = 2
EOS_TOKEN = 3
MAX_LENGTH = 50


class Encoder(nn.Module):
  pass


class Attention(nn.Module):
  pass


class Decoder(nn.Module):
  pass

class Seq2Seq(nn.Module):
  def __init__(self):
    super(Seq2Seq, self).__init__()

