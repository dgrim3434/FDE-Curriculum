from sampler import Sampler
import pytest
import numpy as np
from vocab_preprocessing import Vocabulary
@pytest.fixture(scope='module')
def vocab():
    
    vocab = Vocabulary("corpus/prose_text.txt")
    vocab.build_vocabulary(100, 5)
    
    return vocab

def test_sampler_matches_distribution(vocab):
    
    frequency = vocab.word_frequencies
    
    sampler = Sampler(frequency)
    sampler.negative_sampling()
    
    d = sampler.negative_draw(200_000, 5).ravel()
    emp = np.bincount(d, minlength=len(frequency)) / d.size
    assert np.allclose(emp, sampler.probabilities, atol=2e-3)
    