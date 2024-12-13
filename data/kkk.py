import pandas as pd
import random

# Exemples d'opinions positives et négatives
positive_reviews = [
    "This movie was amazing! Highly recommend it.",
    "Fantastic storyline and brilliant acting.",
    "A truly breathtaking experience, loved every moment!",
    "Incredible visuals and a compelling story.",
    "Highly entertaining and beautifully made.",
    "This film exceeded all my expectations!",
    "A masterpiece in every sense of the word.",
    "Wonderful acting and an unforgettable plot.",
    "This movie is a must-watch for everyone.",
    "I enjoyed every second of this incredible movie."
]

negative_reviews = [
    "The plot was dull and the characters were uninteresting.",
    "I wouldn't watch it again. Pretty boring.",
    "Terrible movie. Waste of time and money.",
    "The acting was subpar, and the story made no sense.",
    "Disappointing and poorly executed.",
    "Not worth the hype, very underwhelming.",
    "I fell asleep halfway through. Absolutely boring.",
    "This movie lacked originality and emotion.",
    "Predictable and utterly forgettable.",
    "A disaster from start to finish."
]

# Générer 1000 lignes d'opinions aléatoires
data = []
for _ in range(1000):
    if random.random() > 0.5:
        data.append((random.choice(positive_reviews), 1))  # Critique positive
    else:
        data.append((random.choice(negative_reviews), 0))  # Critique négative

# Convertir en DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

# Sauvegarder dans un fichier CSV
df.to_csv("movie_reviews_1000.csv", index=False)
print("Fichier 'movie_reviews_1000.csv' généré avec succès !")
