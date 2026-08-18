import pandas as pd
from collections import Counter
import pickle

CAPTION_FILE = "dataset/captions/cleaned_captions.csv"

# Load captions
df = pd.read_csv(CAPTION_FILE)

# Count words
counter = Counter()

for caption in df["caption"]:
    words = caption.split()
    counter.update(words)

# Special tokens
word_to_index = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "<unk>": 3
}

# Add words to vocabulary
for word, count in counter.items():
    if word not in word_to_index:
        word_to_index[word] = len(word_to_index)

# Reverse mapping
index_to_word = {
    index: word
    for word, index in word_to_index.items()
}

print("Vocabulary size:", len(word_to_index))

print("\nFirst 20 words:")

for word, index in list(word_to_index.items())[:20]:
    print(word, "->", index)

# Save vocabulary
with open("dataset/captions/word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

with open("dataset/captions/index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)

print("\n✅ Vocabulary saved!")