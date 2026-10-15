Getting the max token length from the hugging face model:
print(model.max_seq_length)

Calculating the tokens per chunk:
n = len(tok(some_text, add_special_tokens=True)["input_ids"])

Measure for a tokenizer the average tokens per word for your corpus. That's how decisions are made not through an estimation. As I have already seen tokens per word is largely dependent on the
domain.

Extracting the character index from tokenizer:
enc = tok(doc_text, add_special_tokens=False, return_offsets_mapping=True)
ids = enc['input_ids'] -> list[int], length n
offsets = enc['offset_mapping'] -> list[(char_start, char_end)], length n

offset[i] are the characters of doc_text that token i came from

Sentence Chunker: Needs smarter sentence breaks. Regex fails often so we are using spaCy en_core_web_sm
```python
import spacy
nlp spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
for sent in doc.sent:
    sent.start_char, sent.end_char, sent.text
```