import string
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download stopwords once
nltk.download('stopwords', quiet=True)

def preprocess_text(text, stemmer, stopwords_set):
    """
    Preprocess a single text: lowercase, remove punctuation, 
    remove stopwords, and apply stemming.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

def main():
    # Load data
    df = pd.read_csv('spam_ham_dataset.csv')
    df['text'] = df['text'].str.replace('\r\n', ' ', regex=False)
    
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    
    # Preprocess all texts using apply (more efficient than loop)
    print("Preprocessing texts...")
    corpus = df['text'].apply(lambda x: preprocess_text(x, stemmer, stopwords_set))
    
    # Vectorize
    print("Vectorizing texts...")
    vectorizer = CountVectorizer(max_features=5000)  # Limit features to reduce dimensionality
    X = vectorizer.fit_transform(corpus).toarray()
    y = df['label_num']
    
    # Train-test split with random state for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training model...")
    clf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=100)
    clf.fit(x_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    y_pred = clf.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Test on a single email
    print("\n" + "="*50)
    print("Testing on sample email:")
    email_to_classify = df['text'].iloc[10]
    print(f"Original email: {email_to_classify[:100]}...")
    
    # Preprocess the email
    email_preprocessed = preprocess_text(email_to_classify, stemmer, stopwords_set)
    
    # Transform and predict
    x_email = vectorizer.transform([email_preprocessed])
    prediction = clf.predict(x_email)[0]
    probability = clf.predict_proba(x_email)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"\nPrediction: {label}")
    print(f"Confidence: {max(probability):.2%}")

if __name__ == "__main__":
    main()