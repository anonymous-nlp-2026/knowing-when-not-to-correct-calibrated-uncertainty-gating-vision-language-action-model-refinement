import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/gemma-2b-tokenizer')
print(f'Tokenizer class: {type(tok).__name__}')
print(f'Default padding_side: {tok.padding_side}')
print(f'pad_token: {repr(tok.pad_token)} (id={tok.pad_token_id})')
print(f'bos_token: {repr(tok.bos_token)} (id={tok.bos_token_id})')
print(f'eos_token: {repr(tok.eos_token)} (id={tok.eos_token_id})')
print(f'add_bos_token: {getattr(tok, "add_bos_token", "N/A")}')
print(f'add_eos_token: {getattr(tok, "add_eos_token", "N/A")}')

# Sample prompt
state_str = "124 124 126 127 128 125 128 129"
prompt = f"Task: put the moka pot on the stove, State: {state_str};\nAction: "
print(f'\nPrompt: {repr(prompt)}')

# Left padding (default)
tok.padding_side = "left"
tokens_left = tok(prompt, return_tensors="pt", padding="max_length", max_length=200, truncation=True)
ids_left = tokens_left["input_ids"][0].tolist()
mask_left = tokens_left["attention_mask"][0].tolist()
n_real_left = sum(mask_left)
print(f'\n--- LEFT padding ---')
print(f'Real tokens: {n_real_left}')
print(f'First 5 ids: {ids_left[:5]}')
print(f'Last 5 ids: {ids_left[-5:]}')
print(f'Non-pad tokens: {[t for t in ids_left if t != 0]}')
print(f'Decoded non-pad: {tok.decode([t for t in ids_left if t != 0])}')

# Right padding
tok.padding_side = "right"
tokens_right = tok(prompt, return_tensors="pt", padding="max_length", max_length=200, truncation=True)
ids_right = tokens_right["input_ids"][0].tolist()
mask_right = tokens_right["attention_mask"][0].tolist()
n_real_right = sum(mask_right)
print(f'\n--- RIGHT padding ---')
print(f'Real tokens: {n_real_right}')
print(f'First 5 ids: {ids_right[:5]}')
print(f'Last 5 ids: {ids_right[-5:]}')
print(f'Non-pad tokens: {[t for t in ids_right if t != 0]}')
print(f'Decoded non-pad: {tok.decode([t for t in ids_right if t != 0])}')

# Compare
print(f'\nSame non-pad tokens: {[t for t in ids_left if t != 0] == [t for t in ids_right if t != 0]}')
print(f'Same real count: {n_real_left == n_real_right}')
