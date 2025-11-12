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

def plot_data_distribution(df):
    """Plot the distribution of spam vs ham emails."""
    plt.figure(figsize=(8, 5))
    counts = df['label_num'].value_counts()
    labels = ['Ham (0)', 'Spam (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Spam vs Ham Emails', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Email Type', fontsize=12)
    
    for i, v in enumerate(counts):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('01_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_data_distribution.png")

def plot_text_length_distribution(df):
    """Plot the distribution of text lengths for spam vs ham."""
    df['text_length'] = df['text'].str.len()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[df['label_num'] == 0]['text_length'], bins=50, alpha=0.7, 
             label='Ham', color='#2ecc71', edgecolor='black')
    plt.hist(df[df['label_num'] == 1]['text_length'], bins=50, alpha=0.7, 
             label='Spam', color='#e74c3c', edgecolor='black')
    plt.xlabel('Text Length (characters)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Text Length Distribution', fontsize=13, fontweight='bold')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='text_length', by='label_num', ax=plt.gca())
    plt.xlabel('Email Type (0=Ham, 1=Spam)', fontsize=11)
    plt.ylabel('Text Length', fontsize=11)
    plt.title('Text Length by Category', fontsize=13, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('02_text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_text_length_distribution.png")

def plot_word_clouds(df):
    """Generate word clouds for spam and ham emails."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Ham word cloud
    ham_text = ' '.join(df[df['label_num'] == 0]['text'].values)
    ham_wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              colormap='Greens').generate(ham_text)
    
    axes[0].imshow(ham_wordcloud, interpolation='bilinear')
    axes[0].set_title('Most Common Words in HAM Emails', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Spam word cloud
    spam_text = ' '.join(df[df['label_num'] == 1]['text'].values)
    spam_wordcloud = WordCloud(width=800, height=400, 
                               background_color='white',
                               colormap='Reds').generate(spam_text)
    
    axes[1].imshow(spam_wordcloud, interpolation='bilinear')
    axes[1].set_title('Most Common Words in SPAM Emails', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('03_word_clouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_word_clouds.png")

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('04_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_confusion_matrix.png")

def plot_feature_importance(clf, vectorizer, top_n=20):
    """Plot top important features."""
    feature_importance = clf.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top N features
    indices = np.argsort(feature_importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = feature_importance[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importance, color='skyblue', edgecolor='black')
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features (Stemmed Words)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('05_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_feature_importance.png")

    """
    Preprocess a single text: lowercase, remove punctuation, 
    remove stopwords, and apply stemming.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

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
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('spam_ham_dataset.csv')
    df['text'] = df['text'].str.replace('\r\n', ' ', regex=False)
    
    # Visualize data distribution
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS - Generating Visualizations...")
    print("="*60)
    plot_data_distribution(df)
    plot_text_length_distribution(df)
    plot_word_clouds(df)
    print("\n✓ All exploratory plots saved successfully!\n")
    
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    
    # Preprocess all texts using apply (more efficient than loop)
    print("\nPreprocessing texts...")
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
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    y_pred = clf.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Visualize results
    print("\nGenerating performance visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(clf, vectorizer, top_n=20)
    print("\n✓ All performance plots saved successfully!\n")
    
    # Test on a single email
    print("\n" + "="*60)
    print("TESTING ON SAMPLE EMAIL")
    print("="*60)
    email_to_classify = df['text'].iloc[10]
    print(f"Original email: {email_to_classify[:150]}...")
    
    # Preprocess the email
    email_preprocessed = preprocess_text(email_to_classify, stemmer, stopwords_set)
    
    # Transform and predict
    x_email = vectorizer.transform([email_preprocessed])
    prediction = clf.predict(x_email)[0]
    probability = clf.predict_proba(x_email)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"\nPrediction: {label}")
    print(f"Confidence: {max(probability):.2%}")
    print("="*60)

if __name__ == "__main__":
    main()