from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

print("Chat template:")
print(tokenizer.chat_template)
print("\nSpecial tokens:")
print(tokenizer.special_tokens_map)
print("\nAll special tokens:")
print(tokenizer.all_special_tokens)
print("\nAll special token IDs:")
print(tokenizer.all_special_ids)