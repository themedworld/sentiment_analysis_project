from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer
model_path = "../models/saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict(text):
    # Tokeniser le texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Obtenir la sortie du modèle
    outputs = model(**inputs)
    logits = outputs.logits
    # Prédire la classe (0 = négatif, 1 = positif)
    prediction = torch.argmax(logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# Exemple de prédiction
text = "not bad "
print(f"Text: {text}")
print(f"Sentiment: {predict(text)}")
