def create_vocab(corpus):
  unique_chars = set()
  for char in corpus:
    unique_chars.add(char)
  token_map = {char:code for code,char in enumerate(unique_chars)}
  char_map = {code:char for char,code in token_map.items()}
  return token_map, char_map
