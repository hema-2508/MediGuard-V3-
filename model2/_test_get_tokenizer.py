from dataset import get_tokenizer

tokenizer = get_tokenizer()
print("get_tokenizer OK:", type(tokenizer).__name__, "is_fast=", getattr(tokenizer, "is_fast", None))
print("vocab size:", tokenizer.vocab_size)
