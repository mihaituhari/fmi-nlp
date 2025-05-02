import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentiment import trainer, train_test

# Forțare CPU pentru model
trainer.model.to(torch.device("cpu"))

# Etichete
label_map = {0: "pessimist", 1: "neutral", 2: "optimist"}

# Predictie
predictions = trainer.predict(train_test["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Acuratețe
accuracy = accuracy_score(true_labels, preds)
print(f"Accuracy: {accuracy:.4f}")

# Raport
report = classification_report(true_labels, preds, target_names=["pessimist", "neutral", "optimist"])
print("Classification Report:")
print(report)

# Matrice de confuzie
conf_matrix = confusion_matrix(true_labels, preds)
print("Confusion Matrix:")
print(conf_matrix)
