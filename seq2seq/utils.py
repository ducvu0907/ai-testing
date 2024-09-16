import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
import spacy
import tqdm
from torch.nn.utils.rnn import pad_sequence
import warnings

device = torch.device("cuda" if torch.cpu.is_available() else "cpu")
dataset = datasets.load_dataset("bentrevett/multi30k")

en_spacy = spacy.load("en_core_web_sm")
de_spacy = spacy.load("de_core_news_sm")

