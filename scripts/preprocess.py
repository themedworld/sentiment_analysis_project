import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    # Charger les données CSV
    df = pd.read_csv(file_path)
    # Diviser les données en ensembles d'entraînement et de test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    return train_texts.tolist(), test_texts.tolist(), train_labels.tolist(), test_labels.tolist()
