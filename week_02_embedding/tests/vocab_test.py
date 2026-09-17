import pytest
from vocab_preprocessing import Vocabulary


@pytest.fixture(scope='module')
def vocab():
    """Created the vocabulary class"""
    v = Vocabulary("./corpus/prose_text.txt")
    v.build_vocabulary(512, 5)

    return v

def test_vocab_id_word_mapping(vocab):
    
    id_to_word = vocab.idx_to_word
    word_to_idx = vocab.word_to_idx
    
    assert all(word_to_idx[id_to_word[i]] == i for i in range(len(id_to_word)))