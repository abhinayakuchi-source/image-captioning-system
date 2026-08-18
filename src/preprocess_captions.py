import pandas as pd
import re

CAPTION_FILE = "dataset/captions/captions.txt"

# Read captions
df = pd.read_csv(
    CAPTION_FILE,
    sep=","
)

df.columns = ["image", "caption"]

print("Original captions:")
print(df.head())

# Clean captions
def clean_caption(caption):
    caption = caption.lower()

    # Remove punctuation
    caption = re.sub(r"[^a-z\s]", "", caption)

    # Remove extra spaces
    caption = re.sub(r"\s+", " ", caption).strip()

    # Add start and end tokens
    caption = "<start> " + caption + " <end>"

    return caption


df["caption"] = df["caption"].apply(clean_caption)

print("\nCleaned captions:")
print(df.head())

# Save cleaned captions
OUTPUT_FILE = "dataset/captions/cleaned_captions.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n✅ Cleaned captions saved to:")
print(OUTPUT_FILE)

print("\nTotal captions:", len(df))