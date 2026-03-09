from minichatgpt.tokenizer import BytePairTokenizer

def test_tokenizer_roundtrip() -> None:
    text = "Hello hello mini chatgpt"
    tokenizer = BytePairTokenizer()
    tokenizer.train(text, vocab_size=260, verbose=False)
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert decoded == text
    assert tokenizer.vocab_size >= 256
