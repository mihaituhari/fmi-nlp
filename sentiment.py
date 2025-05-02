# Sentiment Analysis: Optimism vs Pessimism Classifier
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- STEP 1: Load cleaned dataset ---
df = pd.read_csv("training_data_cleaned.csv")

# --- STEP 2: Preprocessing ---
label_map = {0: "pessimist", 1: "neutral", 2: "optimist"}
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df[["text", "label"]])
hf_dataset = hf_dataset.map(tokenize, batched=True)
hf_dataset = hf_dataset.rename_column("label", "labels")
hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset
train_test = hf_dataset.train_test_split(test_size=0.3, seed=42)

# --- STEP 3: Define Model ---
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# --- STEP 4: Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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


# --- STEP 8: Predict single sentence ---
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


# Example usage for multiple sentences
example_texts = [
    "Everything is just the way it’s supposed to be.",
    "At least it's raining, so no one can see me cry.",
    "I'm so happy I get to redo everything because of someone else's mistake. Yay.",
    "I lost the job I hated. Finally, I’m free to do something I actually care about."
]

print("\n--- Example Predictions ---")
for text in example_texts:
    label, scores = predict_sentence(text)
    print(f"\nText: '{text}'")
    print(f"Predicted label: {label}")
    print(f"Probabilities: pessimist={scores[0]:.2f}, optimist={scores[2]:.2f}, neutral={scores[1]:.2f}")

# Example usage for single sentence
# example_text = "This is amazingly stupid from your side. i am really happy you will lose and die most probably because you are so stupid. god, i wish you were not my brother"
# label, scores = predict_sentence(example_text)
# print("--- Example Prediction ---")
# print(f"Text: '{example_text}'")
# print(f"Predicted label: {label}")
# print(f"Probabilities: pessimist={scores[0]:.2f}, optimist={scores[2]:.2f}, neutral={scores[1]:.2f}")
