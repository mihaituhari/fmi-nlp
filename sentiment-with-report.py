# Sentiment Analysis: Optimism vs Pessimism Classifier
import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- STEP 1: Load or create dataset ---
data = {
    "text": [
        "Life is beautiful and full of hope!",
        "Everything is pointless and I feel lost.",
        "I am excited for the future! It is great to be alive.",
        "Today is a terrible day.",
        "Not sure how to feel about this week.",
        "The world is amazing and I love being alive.",
        "Nothing makes sense anymore.",
        "I feel optimistic about the changes ahead.",
        "It’s all going to fall apart soon.",
        "I’m doing okay, could be worse.",
        "I love sunny days and good vibes.",
        "I'm feeling down and nothing helps.",
        "Maybe things will get better soon.",
        "This is the worst week of my life.",
        "Today is just another day.",
        "This is good",
        "Great news buddy!"
    ],
    # 1=optimist, 0=pesimist, 2=neutral
    "label": [1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 1]
}

df = pd.DataFrame(data)

# --- STEP 2: Preprocessing ---
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)

label_map = {0: "pessimist", 1: "optimist", 2: "neutral"}


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset
train_test = dataset.train_test_split(test_size=0.3, seed=42)

# --- STEP 3: Define Model ---
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# --- STEP 4: Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# --- STEP 5: Define Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer
)

# --- STEP 6: Train Model ---
trainer.train()

# --- STEP 7: Evaluation ---
predictions = trainer.predict(train_test["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Ensure correct mapping for target names
unique_labels = sorted(list(set(true_labels) | set(preds)))
target_names = [label_map[label] for label in unique_labels]

# --- STEP 8: Report ---
print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))
cm = confusion_matrix(true_labels, preds, labels=unique_labels)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output.png", dpi=300)


# plt.show()


# --- STEP 9: Predict single sentence ---
def predict_sentence(text: str):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    prediction = np.argmax(probs)
    label = label_map[prediction]
    return label, probs


# -- EXAMPLE USAGE --
example_text = "I think tomorrow will be a great day!"
label, scores = predict_sentence(example_text)
print(f"Text: '{example_text}'")
print(f"Predicted label: {label}")
print(f"Probabilities: pessimist={scores[0]:.2f}, optimist={scores[1]:.2f}, neutral={scores[2]:.2f}")
