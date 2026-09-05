GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

def get_stats(arr):
    
    # Logs the count of each match
    matches = {}
    # Tracks the current match leader
    max_pair = None

    # Loops through all words in the array
    for word in arr:
      
      i = 0
      # Loops through all letters within the word
      while i + 1 < len(word):
        # Creates tuple for the character sequence
        pair = (word[i], word[i + 1])

        if pair not in matches:
          matches[pair] = 0

        # Checks to see if it's the max
        if max_pair not in matches or matches[max_pair] < matches[pair] + 1:
          max_pair = pair

        # Increments the count
        matches[pair] += 1

        i += 1

    return max_pair, matches[max_pair], matches

def merge(pair, arr, match_fill):
    # Updated Merged array
    new_arr = []
    i = 0

    # Loops through all characters within the word
    while i < len(arr):

      if arr[i] != pair[0] or i + 1 >= len(arr) or arr[i + 1] != pair[1]:
        new_arr.append(arr[i])
        i += 1
      else:
        new_arr.append(match_fill)
        i += 2
    
    return new_arr
  
class Tokenizer:

  def __init__(self, SPLIT_PATTERN = GPT2_SPLIT_PATTERN, verbose=False):

    self.SPLIT_PATTERN = SPLIT_PATTERN
    self.vocab = {i: bytes([i]) for i in range(256)}
    self.merged = {}
    self.verbose = verbose
    self.vocab_size = None
    self.special_tokens = {} # -> str to id

  def train(self, vocab_size, corpus):
    
    self.vocab_size = vocab_size
    words = re.findall(self.SPLIT_PATTERN, corpus)
    training_words = [chunk.encode('utf-8') for chunk in words]

    assert vocab_size > 256

    iterations = vocab_size - 256

    for i in range(iterations):
      curr_id = i + 256

      max_pair, matches, match_dict = get_stats(training_words)

      if matches < 2:
        print("Training Corpus is to small for the provided Vocab size")
        break

      self.merged[max_pair] = curr_id

      self.vocab[curr_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

      new_training_words = [merge(max_pair, chunk, curr_id) for chunk in training_words]

      assert sum(map(len, new_training_words)) < sum(map(len, training_words)), \
        f"merge {max_pair} was no-op - stale corpus state"
      
      training_words = new_training_words

      assert self.vocab[curr_id] == self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

      if self.verbose:
        print(f"Iteration: {i} merged pair: {max_pair}, total_count: {matches}")

  def decode(self, id_list):

    # Flipping special tokens
    special = {v: k for k,v in self.special_tokens.items()}

    parts = []
    
    for idx in id_list:
      
      if idx == -1:
        parts.append(''.encode('utf-8'))
      elif idx in self.vocab:
        parts.append(self.vocab[idx])
      elif idx in special:
        parts.append(special[idx].encode('utf-8'))
      else:
        raise ValueError(f"Invalid ID: {idx}")
      
    return b"".join(parts).decode('utf-8', errors='replace')
    

    """
    Encoding Section:
      1. Checks for error raises to see how special tokens should be handled
      2. Breaks the input into individual words
      3. Each individual work then gets encoded
      4. the encodings of each of these words are then merged based on the 
      order from training
      5. When you reach the point where you have only 1 chunk then you have fully
      merged
      6. If the returned merge isn't within the merged dictionary you are also complete
      7. The final return is a list of all the ids which make up the input. The lower the ID
      count the better the sentence was encoded
    """
  def encode_chunk(self, chunk):
    # Chunks starts at the lowest possible id level. chunk[i] == character id
    # We then merge ids based on the merges we have seen. self.merge[] == {(id, id) -> id
    # We are checking for possible merges if there are less than 2 you can't merge
    ids = list(chunk)
    while len(ids) > 1:
        
      a,b,stats = get_stats([ids])
      pair = min(stats, key=lambda p: self.merged.get(p, float('inf')))
      if pair not in self.merged:
        break
        
      ids = merge(pair, ids, self.merged.get(pair))
      
    return ids

  def encode_text(self, text):
    ids = []

    if len(re.findall(self.SPLIT_PATTERN, text)) == 0:
      return [-1]
    for chunk in re.findall(self.SPLIT_PATTERN, text):
      ids.extend(self.encode_chunk(chunk.encode('utf-8')))
      
    return ids
    
  # allowed_special = ['all', 'none', 'none_raise']
  def encode(self, text, allowed_special='none_raise'):
      
    if allowed_special == 'all':
      special = self.special_tokens
    elif allowed_special == 'none':
      special = {}
    elif allowed_special == 'none_raise':
      special = {}
      assert all(tok not in text for tok in self.special_tokens)
    else:
      raise ValueError(allowed_special)
      
    if not special:
      return self.encode_text(text)
      
    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"

    ids = []

    for word in re.split(special_pattern, text):

      if word in special:
        ids.append(special[word])
      else:
        ids.extend(self.encode_text(word))
      
    return ids
    
  # Saving the Model (used AI)
  def save(self, file_prefix):
    with open(file_prefix + ".model", "w", encoding="utf-8") as f:
        f.write("bpe-v1\n")
        f.write(f"{self.SPLIT_PATTERN}\n")
        f.write(f"{len(self.special_tokens)}\n")
        for tok, idx in self.special_tokens.items():
            f.write(f"{tok} {idx}\n")
        for (p0, p1) in self.merged:
            f.write(f"{p0} {p1}\n")
    
        
    with open(file_prefix + ".vocab", 'w', encoding='utf-8') as f:
      for idx, tb in self.vocab.items():
        f.write(f"[{tb.decode('utf-8', errors='replace')}] {idx}\n")
    
  def load(self, model_file):
    merges, idx = {}, 256

    with open(model_file, 'r', encoding="utf-8") as f:
      assert f.readline().strip() == "bpe-v1"
      self.SPLIT_PATTERN = f.readline().rstrip("\n")
      special_tokens = {}

      for _ in range(int(f.readline().strip())):
        tok, tok_id = f.readline().strip().split()
        special_tokens[tok] = int(tok_id)
        
      for line in f:
        p0, p1 = map(int, line.split())
        merges[(p0, p1)] = idx
        idx +=1
        
    self.merged = merges
    self.special_tokens = special_tokens
    self.vocab = self._build_vocab()
      
  def _build_vocab(self):
    vocab = {i: bytes([i]) for i in range(256)}

    for (p0, p1), idx in self.merged.items():
      vocab[idx] = vocab[p0] + vocab[p1]
      
    for tok, idx in self.special_tokens.items():
      vocab[idx] = tok.encode('utf-8')
      
    return vocab

  def add_special(self, tokens):
    
    if self.vocab_size is None:
      raise ValueError("You Must train the tokenizer before inserting special tokens")
    
    curr_id = self.vocab_size
    
    for token in tokens:
      curr_id += 1
      self.special_tokens[token] = curr_id
      self.vocab[curr_id] = token.encode('utf-8')
      

