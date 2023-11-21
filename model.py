import pdb
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Model(nn.Module):

  def __init__(self, vocab_size, context_size=64, hidden_dim=128, layers=4):
    super().__init__()
    self.context_size = context_size
    self.token_embed = nn.Embedding(vocab_size, hidden_dim)
    self.position_embed = nn.Embedding(context_size, hidden_dim)
    self.layers = [nn.Linear(context_size*hidden_dim, context_size*hidden_dim) for x in range(layers)]
    self.out_proj = nn.Linear(context_size*hidden_dim, vocab_size)

  def forward(self, token_batch):
    pos = self.position_embed(torch.arange(self.context_size))
    x = self.token_embed(token_batch)# + pos
    x = rearrange(x, "b t c -> b (t c)")
    #x = x.mean(axis=1)
    #pdb.set_trace()
    for layer in self.layers:
      x = x + F.relu(layer(x))
    return self.out_proj(x)
