import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

print("Loading dataset...")
dataset = pd.read_csv('Consumer_Complaints.csv', engine='python', on_bad_lines='skip')

# Simple preprocessing
def preprocess(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower()

# Response mapping
response_mapping = {
    'Closed with explanation': 1,
    'Closed with non-monetary relief': 0,
    'Closed with relief': 1,
    'Closed with monetary relief': 1,
    'Untimely response': 0,
    'Closed ': 1,
    'Closed without relief': 0
}

# Preprocess the data
print("Preprocessing data...")
dataset = dataset[['Company response to consumer', 'Consumer complaint narrative']].copy()
dataset['is_solved'] = dataset['Company response to consumer'].map(response_mapping)
dataset = dataset.dropna(subset=['is_solved', 'Consumer complaint narrative'])
dataset['cleaned_text'] = dataset['Consumer complaint narrative'].apply(preprocess)

# Prepare features and target
X = dataset['cleaned_text']
y = dataset['is_solved'].values

print(f"Training on {len(X)} samples...")
print(f"Class distribution: {pd.Series(y).value_counts()}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X).toarray()

# Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_tfidf, y)

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model and vectorizer saved successfully!")
print(f"Model accuracy: {model.score(X_tfidf, y):.2f}")