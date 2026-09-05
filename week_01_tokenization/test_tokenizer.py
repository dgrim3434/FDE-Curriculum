from tokenizer import Tokenizer, merge
import pytest
class test_tokenizer:
    
    def __init__(self):
        
        self.tokenizer = Tokenizer()
        
        text = """
        Hello, My name is Dylan Grim. I have recently decided to take things to the extreme.
        I am working on becoming the best FDE in the game and that is going to take a lot of work
        but I am completely here for it and am not scared about anything since God has my back and
        will continue to guide me. This is my training Corpus which I will use to test my tokenizer for
        week one.
        """
        
        self.tokenizer.train(512, text)
    
    def test_merges(self):
        assert merge((1,1), [1,1,1], 99) == [99,1]
        assert merge([1,2], [1,2], 1) == [1]
        assert merge([5, 6], [1,2,3], 99) == [1,2,3]
        assert merge([1,2], [1], 99) == [1]
  
        
  
    def test_roundtrip_property(self):
        import random

        for _ in range(1000):

            n = random.randint(0, 200)
            s = "".join(chr(random.randint(0, 0x10FFFF)) for _ in range(n))
            s = "".join(c for c in s if not (0xD800 <= ord(c) <= 0xDFFF))

            assert self.tokenizer.decode(self.tokenizer.encode(s)) == s

    def test_edge_cases(self):
        for s in ["", " ", "  ", "    ", "\n", "\t\t", "a", "aaaaaaaa",
                "é", "中文测试", "🙂", "👨‍👩‍👧", "café",
                "def f(x):\n    return x", "1234567890", "a" * 100000]:
        
        
            assert self.tokenizer.decode(self.tokenizer.encode(s)) == s, repr(s)
  
    def test_save_load_identity(self):
        self.tokenizer.save("models/test")
        tok2 = Tokenizer(); tok2.load("models/test.model")
        assert tok2.merged == self.tokenizer.merged
        assert tok2.vocab  == self.tokenizer.vocab
        sample = "Hello My Name is Dylan"
        assert tok2.encode(sample) == self.tokenizer.encode(sample)
    def test_vocab_merges_consistency(self):
        for (p0, p1), idx in self.tokenizer.merged.items():
            assert self.tokenizer.vocab[idx] == self.tokenizer.vocab[p0] + self.tokenizer.vocab[p1]
    
    def test_special_tokens(self):
        
        self.tokenizer.add_special(['<|endoftext|>', '<|im_start|>', '<|im_end|>'])
        
        text = "<|im_start|>Hi My Name Is dylan and I am using special Tokens within my prompt<|im_end|>"
        
        encoding = self.tokenizer.encode(text, allowed_special='all')
        encoding2 = self.tokenizer.encode(text, allowed_special='none')
        
        assert text == self.tokenizer.decode(encoding)
        assert text == self.tokenizer.decode(encoding2)
        
        with pytest.raises(AssertionError):
            encoding2 = self.tokenizer.encode(text, allowed_special='none_raise')
        # print(f"Encoding 1: {encoding}")
        # print(self.tokenizer.decode(encoding))
        # print(f"Encoding 2: {encoding2}")
        # print(self.tokenizer)
        
if __name__ == "__main__":
    
    test = test_tokenizer()
    
    try:
        test.test_merges()
        print("Passed Merge Test")
    except AssertionError as e:
        print("Failed Merge Test")
    
    try:
        test.test_roundtrip_property()
        print("Passed Round Trip")
    except AssertionError as e:
        print("Failed Round Trip Property")
    
    try:
        test.test_edge_cases()
        print("Passed Edge Cases")
    except AssertionError as e:
        print("Failed Edge Cases")
    
    try:
        test.test_save_load_identity()
        print("Passed Save Load Test ")
    except AssertionError as e:
        print("Failed Save Load Test:", e)

    try:
        test.test_vocab_merges_consistency()
        print("Passed merge Consistency Test")
    except AssertionError:
        print("Failed merge Consistency Test")
    
    try:
        
        test.test_special_tokens()
        print("Passed the Special Tokens Test")
    except :
        print("Failed Special Tokens Test")
    
    
    