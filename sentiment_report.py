import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentiment import trainer, train_test

label_map = {0: "pessimist", 1: "neutral", 2: "optimist"}

predictions = trainer.predict(train_test["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

accuracy = accuracy_score(true_labels, preds)
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(true_labels, preds, target_names=["pessimist", "neutral", "optimist"])
print("Classification Report:")
print(report)

conf_matrix = confusion_matrix(true_labels, preds)
print("Confusion Matrix:")
print(conf_matrix)
