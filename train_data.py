import html
import string
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import re
from datasets import load_dataset

print("🔁 Loading TweetEval sentiment dataset...")
dataset = load_dataset("tweet_eval", "sentiment")
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()
df_val = dataset["validation"].to_pandas()
df = pd.concat([df_train, df_test, df_val], ignore_index=True)
df = df.sample(50, random_state=42)
df.to_csv("data/training.csv", index=False)
print("✅ Exported initial training data: data/training.csv")

label_map = {0: "pessimist", 1: "neutral", 2: "optimist"}
df["label_name"] = df["label"].map(label_map)

def clean_text(text):
    text = html.unescape(text)

    # Elimină @user
    text = re.sub(r'@\w+', '', text)

    # Elimină URL-uri
    text = re.sub(r'http\S+', '', text)

    # Elimină emoji-urile
    text = re.sub(r'[^\w\s' + re.escape(string.punctuation) + ']', '', text)

    # Elimină caracterele non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Elimină semnele de punctuație repetitive excesive, înafara de ?!
    text = re.sub(r'[^\w\s!?]', '', text)

    # Elimină semnele de punctuație repetitive"
    text = re.sub(r'([!?]){2,}', r'\1', text)
    text = text.lower()

    # Înlocuiește spațiile multiple cu un singur spațiu
    text = re.sub(r'\s+', ' ', text)

    # Înlocuiește cuvintele abreviate
    text = text.replace('gr8', 'great').replace('b4', 'before').replace('luv', 'love')

    return text

df["text"] = df["text"].apply(clean_text)
df.to_csv("data/training_cleaned.csv", index=False)

print("✅ Exported cleaned training data/training_cleaned.csv")
