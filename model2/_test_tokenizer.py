from transformers import AutoTokenizer, BertTokenizer

for label, path in [
    ("Auto hub", "dmis-lab/biobert-base-cased-v1.1"),
    ("Auto ckpt", "checkpoints/best_model"),
    ("Auto cache", "cache/tokenizer"),
]:
    try:
        t = AutoTokenizer.from_pretrained(path)
        print(f"{label} OK {type(t).__name__}")
    except Exception as e:
        print(f"{label} FAIL: {e}")

try:
    t = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)
    print("Auto hub use_fast=True OK")
except Exception as e:
    print(f"Auto hub use_fast=True FAIL: {e}")

try:
    t = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=False)
    print("Auto hub use_fast=False OK")
except Exception as e:
    print(f"Auto hub use_fast=False FAIL: {e}")

try:
    t = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    print(f"BertTokenizer OK {type(t).__name__}")
except Exception as e:
    print(f"BertTokenizer FAIL: {e}")

try:
    t = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", tokenizer_type="bert")
    print(f"Auto tokenizer_type=bert OK {type(t).__name__}")
except Exception as e:
    print(f"Auto tokenizer_type=bert FAIL: {e}")
