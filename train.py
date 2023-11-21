import pdb
import random
import torch
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
from model import Model
from tokenizing import create_vocab

def main():
  
  print("building vocab")

  with open("data/all_notes.txt", "r") as f:
    full_corpus = f.read()
  
  token_map, char_map = create_vocab(full_corpus)

  print("loading splits")

  with open("data/train_split.txt", "r") as f:
    train_corpus = f.read()
  
  with open("data/test_split.txt", "r") as f:
    test_corpus = f.read()
 
  device = "cpu"  #"cuda" # cpu
  
  context_size = 8 #16 #64
  hidden_dim = 16 #128
  vocab_size = len(token_map.keys())
 
  print("initializing model")

  model = Model(vocab_size, context_size=context_size, hidden_dim=hidden_dim, layers=2).to(device)

  iterations = 100
  steps_per_iter = 20000 #10000
  test_steps = 100
  batch_size = 64
  
  opt = Adam(model.parameters())
  
  with torch.no_grad():
    init_inp, init_out = get_batch(test_corpus, token_map, batch_size, context_size, device)
    print(f"initial sample: \n{generate_sample(model, init_inp[0:1], char_map)}")

  print("begin training")

  for i in range(iterations):
    mean_train_loss = 0
    for e in range(steps_per_iter):
      input_t, output_t = get_batch(train_corpus, token_map, batch_size, context_size, device)
      opt.zero_grad()
      logits = model(input_t)
      # probs = F.softmax(logits, dim=1)
      #if i == 5:
      #  pdb.set_trace()
      loss = F.cross_entropy(logits, output_t)
      mean_train_loss += loss.cpu().item()
      loss.backward()
      opt.step()
    
    mean_train_loss /= steps_per_iter
    print(f"train loss: {mean_train_loss}")
    eval_in, eval_out = get_batch(test_corpus, token_map, batch_size, context_size, device)
    with torch.no_grad():
      mean_test_loss = 0
      for tb in range(test_steps):
        input_t, output_t = get_batch(test_corpus, token_map, batch_size, context_size, device)
        logits = model(input_t)
        loss = F.cross_entropy(logits, output_t)
        mean_test_loss += loss.cpu().item()
      mean_test_loss /= test_steps
      print(f"test loss: {mean_test_loss}")
      sampled_text = generate_sample(model, eval_in[0:1], char_map)
      print(f"{sampled_text}")

def generate_sample(model, curr_context, char_map):
  raw_chars = [char_map[t.item()] for t in curr_context[0][:16]]
  new_text = ""
  for c in raw_chars:
    new_text += c
  new_text += "|generation-start|"
  for i in range(100):
    logits = model(curr_context)
    #probs = F.softmax(logits, dim=1)[0]
    new_token = torch.distributions.Categorical(logits=logits).sample()
    new_text += char_map[new_token.item()]
    curr_context[0] = curr_context[0].roll(-1)
    curr_context[0][-1] = new_token
    #pdb.set_trace()

  return new_text + "|generation-end"
  
def get_batch(data, token_map, batch_size, context_size, device):
  max_idx = len(data) - context_size - 1
  offsets = [random.randint(0, max_idx) for _ in range(batch_size)]
  input_tokens = torch.tensor([[token_map[c] for c in data[off:off+context_size]] for off in offsets], device=device)
  #print(f"input chars : {input_tokens}")
  output_tokens = torch.tensor([token_map[data[off+context_size]] for off in offsets], device=device)
  #print(f"next chars : {output_tokens}")
  return input_tokens, output_tokens

if __name__ == "__main__":
  main()
