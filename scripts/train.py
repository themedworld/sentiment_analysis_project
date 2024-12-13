from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from preprocess import load_and_split_data

# Charger et préparer les données
file_path = "C:/Users/XPS/Desktop/Nouveau dossier (7)/deeplearning/sentiment_analysis_project/data/imdb_reviews.csv"
train_texts, test_texts, train_labels, test_labels = load_and_split_data(file_path)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Préparer les données pour Hugging Face Dataset
def tokenize_data(texts, labels):
    tokens = tokenizer(texts, truncation=True, padding=True, max_length=128)
    tokens["labels"] = labels
    return tokens

train_dataset = Dataset.from_dict(tokenize_data(train_texts, train_labels))
test_dataset = Dataset.from_dict(tokenize_data(test_texts, test_labels))

# Charger le modèle
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="../models/saved_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
)

# Entraîner le modèle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model("../models/saved_model")
tokenizer.save_pretrained("../models/saved_model")
