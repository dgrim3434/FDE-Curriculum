# Week 1 - BPE Tokenizer from Scratch

A byte-level Byte pair tokenizer fully from scratch in python, with no tokenizer libraries used in the core algorithm.

**Repo:** https://github.com/dgrim3434/FDE-Curriculum/tree/main/week_01_tokenization

## What I Built

A Full Byte level tokenizer, with a test suit which tests and compares the tokenizers performance against OpenAI's toktoken API. 4 models where saved and compared against toktoken these tokenizers where trained using two different Corpuses and vocabulary sizes. The two corpuses where the python standard library and the moby dick book which was pulled from the internet. Two tokenizers where trained for each corpus using Vocab size 1024 and 4096 to compare the compression rates for these vocabulary sizes.

**Components:**

|File | Purpose|
| 'tokenizer.py'| 'Tokenizer Class' - 'train' / 'Encode' / 'Decode' / 'Save' / 'Load'|
| 'corpus_creation'| Train/test splits with leakage assertions for both Corpuses|
| 'tokenizer_training.py'| Trains 4 tokenizers at 2 Vocabulary sizes|
| 'compare.py' | Compares tokenizers against tiktoken on held-out text|
| 'test_tokenizer.py' | Tests the tokenizer Class against common test cases to ensure the class is correct|
| 'train_demo' | Runs the tokenizer on the full python Corpus and logs the Vocabulary|
| 'generate_golden.py' & 'test_golden.py'| Compares the tokenizer against the golden standard to ensure version control|

## Architecture

Here is an outline of the full architecture within the Tokenizer Class.

### Byte-level base vocabulary

256 byte values where used instead of character-level base because we wanted to ensure that all characters where accounted for including special characters. The 256 byte values ensures that all results can be represented so the output will not have any <UNK> values which is the main issue with word level vocabularies. It also solves the issue of character-level bases which result in extremely long sequences and force the model to learn common character sequences on it's own. The 256 byte values allow for learning the most common character sequences and allow for intellegent merges across sequences. Lastly this also ensures the round trip test to hold true for all character inputs including special characters and foreign languages.

### Regex pre-tokenization

Before tokenization the input string is first passed through the GPT-2 regex pattern. This converts the string into an array of words which is extremely important because merges should only happen within a specific word instead of across words. This ensures the tokenizer generalizes well to new inputs and maximizes the likelihood that common sequences are represented. An example really helps explain the importance. Take this input string "Hello my Name is Dylan" if the tokenizer merges across words then a token could be "Hello m" this token will not be able to represent when we see "Hello" and will only cover the case where hello is directly followed by the letter 'm' which really doesn't hold much meaning. Allowing merges across words will likely result in higher compression rates within the training set but will destroy the results when used on text which the tokenizer was not trained on.

### Rank-based encoding

This is one of the most important peices of the tokenizer when it comes to encoding text. Iterating over the sequence instead of iterating through the merge list is much more efficient and ensures the encoding process follows the exact same steps as the tokenizer was trained on. This is done by breaking the test into all possible merges and looking them up within the merge list and using the merge with the lowest rank because that was the first merge which was learned by the tokenizer. All later merges rely on the merges which came before it to be complete so this is a must.

Here are the lines of code which ensure this structure:

```python

pair = min(pairs, key=lambda p: self.merged.get(p, float('inf')))

if pair not in self.merged:
    break
```

This code ensures that merges happen in the same order as the tokenizer learned then and is why order is extremely important within the merged list. If the merge isn't within the list infinity will be returned. This ensures that the minimum will always be returned no matter how large the vocabulary is. When no more merges exist you have reached the final state for the word and the next pair will not exist within the merged list which is why we must check to ensure the pair exists before merging.

### Vocabulary as 'int' -> 'bytes'

The vocabulary dictionary is built with the key representing the ID and the value being the bytes which encode the character sequence. The value must be bytes instead of a string because the decoder takes bytes not strings and will throw an error if you try to decode bytes. When decoding you must concatinate the ID's before decoding because depending on the splits and characters which where encoded it might be represented across multiple bytes which must be concatinated before being decoded to ensure the input matches the output. This is the main thing which the round trip test: decode(encode(s)) == s check to ensure. For cases where you are streaming results or extracting chunks of text through RAG there are cases where your chunk splits between bytes of a character which will produce invalid results. For this reason when decoding you must include errors='replace' which will ensure errors will not be raised and the results will show as a '?' instead of an error.

### Special token policy

Special tokens are extremely important later on when assigning roles for the agents and adding system prompts. Because these special tokens must be extracted from the input to ensure the system sees them they must be handled with care. This opens the system up to user attackers where they can use prompt injection in order to get the agent to work in ways which it wasn't intended. Because of this there are three ways which the special tokens can be handled:

    'all': means the special tokens will be extracted and passed into the agent which will allow for roles and system prompts to be defined. This should never be used when handling user input as it will allow for prompt injection

    'none': means the special tokens will be encoded as regular text. The text will be processed but will be treated as regular tokens which will not affect the agent.

    'none_raise': If special tokens are present within the prompt an exception will be raised and the process will terminate.

Since special tokens are extremely important you can't always set 'allowed_special' to 'none' or 'none_raise' because this will ruin system prompts and role assignments. Because of this you must encode each peice of the prompt seperately which means breaking into system prompt + user_input + closing_prompt. Where special tokens are allowed for both the system and closing prompt but turned off for user_input.

prompt = encode(system_prompt, allow_special = 'all') + encode(user_prompt, allow_special = 'none') + encode(closing_prompt, allow_special = 'all')

This setup ensures the system prompts work as designed and ensure the user can not inject any unsafe instructions which might cause the agent to work in a way which it's not supposed to.

### How BPE learns Merges

Within the BPE tokenizer a merge is selected by finding which sequence of bytes occur most often within the training corpus within a word. The within a word peice is extremely import the tokeinzer only looks at sequences which occur within a single word and not across words which is extremely important when it comes time for the tokenizer to generalize to text it hasn't seen. When a merge is selected that means the sequence of bytes will now become a token and thus must receive an ID and be added to the vocab dictionary. The next step is updating the corpus so that every time that sequence of bytes is seen it is now replaced with the new ID. This is an extremely important step because all future merges rely on the fact that the corpus has been updated after the last merge was completed. This process continue until we have reached the token limit.
In the case of ties it's important that the selection is deterministic meaning if the tokenizer is trained on the same data it will take the same exact merge at each step. When updating the corpus 

### Serialization format

Plain text was chosen over JSON because JSON requires that keys must be strings which is violated within our design since we use tuples to represent merges. In order to correct this we would have to encode the ids as strings which is extra work and could lead to downstream errors. Plain text also ensures that ordering is preserved by convention. Using JSON means that users loading the file must also ensure they are using the correct packages to ensure order is preserved which adds additional overhead which can be avoided by using a simple text file. Within the text file the ordering is equal to the line number and it's extremely easy to visualize the ordering in a very simple manner.

## Code

**Repo:** [(https://github.com/dgrim3434/FDE-Curriculum/tree/main/week_01_tokenization)]

```python
def merge(pair, arr, match_fill):
    new_arr = []
    i = 0

    while i < len(arr):

      if arr[i] != pair[0] or i + 1 >= len(arr) or arr[i + 1] != pair[1]:
        new_arr.append(arr[i])
        i += 1
      else:
        new_arr.append(match_fill)
        i += 2
    
    return new_arr

def get_stats(arr):
    
    matches = {}
    max_pair = None

    for word in arr:
      
      i = 0
      while i + 1 < len(word):
        pair = (word[i], word[i + 1])

        if pair not in matches:
          matches[pair] = 0

        if max_pair not in matches or matches[max_pair] < matches[pair] + 1:
          max_pair = pair

        matches[pair] += 1

        i += 1

    return max_pair, matches[max_pair], matches
    ...
```


## Demo

### Training Logs

IDs: 32 32, Vocab: b'  ', Decoding: '  '
IDs: 256 256, Vocab: b'    ', Decoding: '    '
IDs: 256 32, Vocab: b'   ', Decoding: '   '
IDs: 10 257, Vocab: b'\n    ', Decoding: '    '
IDs: 115 101, Vocab: b'se', Decoding: 'se'
IDs: 32 105, Vocab: b' i', Decoding: ' i'
IDs: 10 258, Vocab: b'\n   ', Decoding: '   '
IDs: 259 258, Vocab: b'\n       ', Decoding: '       '
IDs: 32 116, Vocab: b' t', Decoding: ' t'
IDs: 111 110, Vocab: b'on', Decoding: 'on'
IDs: 114 101, Vocab: b're', Decoding: 're'
IDs: 259 257, Vocab: b'\n        ', Decoding: '        '
IDs: 105 110, Vocab: b'in', Decoding: 'in'
IDs: 116 101, Vocab: b'te', Decoding: 'te'

### Round-trip and edge cases

Passed Merge Test
Passed Round Trip
Passed Edge Cases
Passed Save Load Test 
Passed merge Consistency Test
Passed the Special Tokens Test

### Comparison against tiktoken

| Tokenizer | V | code c/t | prose c/t | ratio |
| mine-code-1024 | 1,024 | 2.37 | 1.88 | 1.26 |
| mine-code-4096 | 4,096 | 2.97 | 2.35 | 1.27 |
| mine-prose-1024 | 1024 | 1.70 | 2.40 | 0.71 |
| mine-prose-4096 | 4,096 | 1.97 | 3.03 | 0.65 |
| gpt2 | 50,257 | 2.18 | 3.70 | 0.59 |
| cl100k_base | 100,277 | 3.91 | 4.04 | 0.97 |
| o200k_base | 200,019 | 3.86 | 4.06 | 0.95 |

These results are extremely telling and show two very important things when working with tokenizers. The first is that when using a smaller vocabulary the training corpus becomes extremely important for the tokenizers ability to generalize well to new text. So if you are trying to save cost and use a smaller vocabulary you must ensure the corpus is a good representation of the text which the tokenizer will see in production. The second peice is that increasing the vocabulary size will improve the compression rate but with diminishing returns. You can see that moving from 50,000 to 100,000 significantly increases the compression rate and allows performance to stabalize across different types of text. Upon further research I found that the regex pattern was changed from gpt2 to the larger vocabulary models with the goal of increasing performance on code based text so the performance increase cannot be soley attributed to the increase in vocabulary size. You can see that when increasing from 100,000 to 200,000 you do not see as much of an increase which shows that there is likely an optimal range for vocabulary size which maximizes the compression and overall efficiency. Another important observation is that both tokenizers trained strictly on python code outperformed the gpt2 tokenizer on python code. This is extremely important because the gpt2 tokenizer has a vocabulary size of 50,000 and both the 1024 and 4096 vocabulary tokenizers trained strictly on python code had higher compressions rates which is a significant finding. The cl100k_base did outperform both of these tokenizers which shows that scale does end up winning in the long run.

### Merge-list diff - Code-trained vs. prose-trained
rank  code-trained            prose-trained           
------------------------------------------------------
0     '  '                    ' t'                    
1     '    '                  'he'                    
2     '   '                   ' a'                    
3     '\n   '                 'in'                    
4     '       '               ' s'                    
5     ' 0'                    ' the'                  
6     'se'                    ' o'                    
7     ' #'                    ' w'                    
8     " '"                    're'                    
9     're'                    'nd'                    
10    'in'                    'ha'                    
11    'le'                    'er'                    
12    'de'                    'ing'                   
13    ' L'                    ' b'                    
14    ' t'                    'ou'                    
15    '\n       '             'hi'                    
16    '\n    '                'ed'                    
17    ' -'                    ' m'                    
18    'on'                    '\n\n'                  
19    'te'                    'it'                    
20    'me'                    'on'                    
21    ' ->'                   ' f'                    
22    "',"                    ' c'                    
23    ' i'                    ' and'                  
24    'or'                    'le'                    
25    ' ='                    ' of'                   
26    '00'                    'or'                    
27    'IN'                    ' to'                   
28    ' a'                    'st'                    
29    ' c'                    'll'                    
30    'ER'                    'en'                    

### Boundary Comparison
'The quick brown fox jumps over the lazy dog.'
  mine-code-4096     (21) The |  qu | i | ck |  b | ro | wn |  f | o | x |   | j | u | mp | s |  over |  the |  lazy |  do | g | .
  mine-prose-4096    (15) The |  quick |  brown |  fo | x |  jum | ps |  over |  the |  l | az | y |  do | g | .
  gpt2               (10) The |  quick |  brown |  fox |  jumps |  over |  the |  lazy |  dog | .
  cl100k_base        (10) The |  quick |  brown |  fox |  jumps |  over |  the |  lazy |  dog | .

'TX-88421-QQ'
  mine-code-4096     ( 8) T | X | - | 88 | 4 | 21 | - | QQ
  mine-prose-4096    (11) T | X | - | 8 | 8 | 4 | 2 | 1 | - | Q | Q
  gpt2               ( 7) TX | - | 884 | 21 | - | Q | Q
  cl100k_base        ( 6) TX | - | 884 | 21 | -Q | Q

'def process_batch(self, x):'
  mine-code-4096     (10) def |  process | _ | b | atch | ( | self | , |  x | ):
  mine-prose-4096    (15) de | f |  pro | cess | _ | b | at | ch | ( | self | , |   | x | ) | :
  gpt2               ( 9) def |  process | _ | batch | ( | self | , |  x | ):
  cl100k_base        ( 7) def |  process | _batch | (self | , |  x | ):

### What I Learned

1. When using tokenizers with smaller vocabulary sizes the corpus is extremely important. From the results you can see the compression rate significantly decreased for both 1024 vocab size tokenizers when tested on text from a different domain.

2. For specific domains like code specialized tokenizers perform significantly better when specific words have significant meaning. When Comparing the boundaries for the text 'def process_batch(self, x):' you see the tokenizer which was trained on code represented 'def' as it's own token where as the prose tokenizer split 'def' into two different tokens 'de' 'f' which will have a significant impact when the LLM tries to process the code. 

3. Increasing the vocabulary size can significantly increase the compression rate up to a certain point and then the returns are less worth it. A tokenizer with vocabulary size 100,000 had very similar performance when compared to a tokenizer with vocabulary size 200,000. Doubling the vocabulary size didn't have an affect on the compression rate but still significantly impacted the performance of the input and output layer

4. The fact that tokenizers struggle to learn how to represent mathematical functions like 50 x 354 could break into multiple different tokens like |5|0| X 
3| 54| making it hard for the LLM to answer the mathematical question. Similar to 'how many r's are in strawberry' when the tokenizer breaks 'strawberry' into 'str' 'aw' 'berry' and it hallucinates on the answer.

5. FDE hack to never end a prompt with whitespace. When predicting the next token for a prompt which ends with whitespace it makes the tokenizer only search for sequences which show up within a word. This is because the Regex splits words so that they maintain the whitespace in front of it. I noticed this when tested where ' Hello' was represented as one token but 'Hello' was represented as two.

6. Due to another error which I was encountering within my Tokenizer which was causing issues when decoding an empty string I had decided to encode an empty string with the ID -1. This is an issue as the tokenizer vobabulary has an invarient that the the range must be from 0 to vocab_size which is an important assumption which if false can have downstream affects. Upon making this change and re-running the test suit I noticed the golden tokenizer test failed since the tokenizer was changed. This is important since even a simple change must be flagged because it can have downstream affects which you might not realize at the time.

## Question

1. My biggest question right now is forward looking for how the LLM is able to accurately answer math questions when the equation is broken into many different tokens. I know the LLM makes use of tools but I am still a little confused about how you know when to call the tools and how you concatinate the tokesn properly to pass them into the tool.

2. The regex seems extremely important and I wonder how much work gets put into developing better regex when in order to split the spacing between words. For tokenizers specializing on code are there certain patters that work better for code and other patters which work better for general text? Are there a couple of best regex patterns or do they largely depend on the use case?

3. How do you come up with the optimal vocabulary size. Is it similar to classical machine learning where you compare the compression rate of a couple of different vocabulary sizes against hold out text and make the decision that way.

4. Within the curriculum this week there where mentions of a different merge strategy which merges based on maximizing the likelihood of the corpus I would assume when solving the likelihood function you would end up with the same results as merging the two most common sequences but if they are different techniques I would assume that's not true.