from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os

class LogisticRegressionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            stop_words=['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc', 'car', 'ni']
        )
        self.model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',  # Enable multi-class classification
            solver='lbfgs'  # Better for multi-class
        )
        # Load and train with data from CSV
        self._load_and_train()

    def _load_and_train(self):
        try:
            # Read the CSV file
            data_path = os.path.join('data', 'processed', 'data.csv')
            df = pd.read_csv(data_path)
            
            # Get unique labels and create label mapping
            unique_labels = sorted(df['label'].unique())
            self.label_map = {i: label for i, label in enumerate(unique_labels)}
            self.reverse_label_map = {label: i for i, label in enumerate(unique_labels)}
            
            # Convert labels to numeric values
            y = df['label'].map(self.reverse_label_map).values
            X = df['text'].values
            
            # Train the model
            self.fit(X, y)
            
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            # Fallback to sample data if CSV loading fails
            self._initialize_with_sample_data()

    def _initialize_with_sample_data(self):
        # Sample training data (fallback)
        X_train = [
            "bonjour comment ça va",
            "je voudrais commander une pizza",
            "quel est le temps",
            "merci beaucoup pour votre aide",
            "je ne comprend pas ce problème",
            "pouvez vous m'aider avec ce projet",
            "bonne journée à vous",
            "ce produit est incroyable",
            "je suis désolé pour le retard",
            "quelle est votre adresse e mail"
        ]
        y_train = [
            "salutation",
            "commande",
            "question",
            "remerciement",
            "confusion",
            "demande",
            "salutation",
            "opinion",
            "excuse",
            "question"
        ]
        
        # Create label mappings
        unique_labels = sorted(set(y_train))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.reverse_label_map = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert labels to numeric values
        y = [self.reverse_label_map[label] for label in y_train]
        self.fit(X_train, y)

    def fit(self, X, y):
        X_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_vect, y)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        # Get probabilities for each class
        probabilities = self.model.predict_proba(X_vect)
        # Get the predicted class
        predictions = self.model.predict(X_vect)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            # Get the probability of the predicted class
            confidence = prob[pred]
            # Get top 3 predictions with their probabilities
            top_indices = np.argsort(prob)[-3:][::-1]  # Get indices of top 3 probabilities
            top_predictions = {
                self.label_map[idx]: round(float(prob[idx]), 2)
                for idx in top_indices
            }
            
            results.append({
                "label": self.label_map[pred],
                "confiance": round(float(confidence), 2),
                "autres_categories": top_predictions
            })
        return results

    def evaluate(self, X, y):
        X_vect = self.vectorizer.transform(X)
        return self.model.score(X_vect, y)