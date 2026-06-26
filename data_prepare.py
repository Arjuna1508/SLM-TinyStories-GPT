import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

print("\nSample stories:")
for i in range(3):
    print("\n--- Story", i + 1, "---")
    print(dataset["train"][i]["text"][:500])

def tokenize_story(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

print("\nTokenizing dataset...")
tokenized = dataset.map(
    tokenize_story,
    remove_columns=["text"],
    desc="Tokenizing stories"
)

for split, filename in [("train", "train.bin"), ("validation", "val.bin")]:
    total_tokens = np.sum(tokenized[split]["len"])
    path = os.path.join("data", filename)

    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    index = 0
    for example in tqdm(tokenized[split], desc=f"Writing {filename}"):
        ids = np.array(example["ids"], dtype=np.uint16)
        arr[index:index + len(ids)] = ids
        index += len(ids)

    arr.flush()
    print(f"{filename} saved with {total_tokens} tokens")

print("\nVerifying saved train.bin...")
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
first_tokens = train_data[:50].tolist()

print("Total train tokens:", len(train_data))
print("First 50 token IDs:")
print(first_tokens)
print("\nDecoded text:")
print(enc.decode(first_tokens))