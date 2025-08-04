from transformers import AutoTokenizer

# Load the tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Discover markdown token IDs
markdown_tokens = {
    "#": tokenizer.encode("#", add_special_tokens=False),
    "##": tokenizer.encode("##", add_special_tokens=False),
    "###": tokenizer.encode("###", add_special_tokens=False),
    "####": tokenizer.encode("####", add_special_tokens=False),
    "**": tokenizer.encode("**", add_special_tokens=False),
    " **": tokenizer.encode(" **", add_special_tokens=False),
    "**:": tokenizer.encode("**:", add_special_tokens=False),
}

for token, ids in markdown_tokens.items():
    print(f"'{token}': {ids}")