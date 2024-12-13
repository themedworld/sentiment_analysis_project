from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Vérifier si le GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger le modèle et le tokenizer
model_path = "../models/saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Déplacer le modèle vers le GPU
model.to(device)

def predict(text):
    # Tokeniser le texte
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Déplacer les entrées vers le GPU
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    # Obtenir la sortie du modèle
    with torch.no_grad():  # Pas besoin de gradients pour les prédictions
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Prédire la classe (0 = négatif, 1 = positif)
    prediction = torch.argmax(logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# Exemple de prédiction
text = " not good "
print(f"Text: {text}")
print(f"Sentiment: {predict(text)}")
