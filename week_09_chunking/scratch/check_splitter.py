import spacy
from chunking.splitter import make_spacy_splitter

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
split = make_spacy_splitter(nlp)
text = "Patients improved (p < 0.05; see Fig. 2). Smith et al. found similar results."
print(split(text))                                  # expect [(0, 41), (42, 77)]
print([text[s:e] for s, e in split(text)])