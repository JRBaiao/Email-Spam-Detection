# 📧 Email Spam Classification System

A machine learning-based spam detection system that uses Natural Language Processing (NLP) and Random Forest classification to accurately identify spam emails.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end spam classification pipeline that processes email text data and distinguishes between legitimate (ham) and spam emails. The system uses advanced NLP techniques for text preprocessing and a Random Forest classifier for prediction.

## ✨ Features

- **Advanced Text Preprocessing**
  - Lowercase conversion and punctuation removal
  - Stopword filtering using NLTK
  - Porter Stemming for word normalization
  - Efficient text vectorization

- **Machine Learning Model**
  - Random Forest Classifier with 100 estimators
  - Stratified train-test split for balanced evaluation
  - Feature importance analysis
  - Probability-based confidence scoring

- **Comprehensive Visualizations**
  - Data distribution analysis
  - Text length comparison (spam vs. ham)
  - Word clouds for visual word frequency
  - Confusion matrix heatmap
  - Top 20 most important features

- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score metrics
  - Detailed classification report
  - Confusion matrix analysis

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JRBaiao/Email-Spam-Detection.git
cd Email-Spam-Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords:
```python
python -c "import nltk; nltk.download('stopwords')"
```

### Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

## 💻 Usage

### Basic Usage

```bash
python spam_classifier.py
```

The script will:
1. Load and preprocess the dataset
2. Generate exploratory data analysis visualizations
3. Train the Random Forest model
4. Evaluate model performance
5. Save all visualizations as PNG files
6. Test prediction on a sample email

### Output Files

After running the script, you'll find these visualizations in your directory:
- `01_data_distribution.png` - Class balance visualization
- `02_text_length_distribution.png` - Text length analysis
- `03_word_clouds.png` - Word frequency visualization
- `04_confusion_matrix.png` - Model performance matrix
- `05_feature_importance.png` - Top predictive features

### Custom Prediction

To classify your own email:

```python
from spam_classifier import preprocess_text, clf, vectorizer, stemmer, stopwords_set

# Your email text
email = "Congratulations! You've won $1000. Click here to claim now!"

# Preprocess and predict
email_preprocessed = preprocess_text(email, stemmer, stopwords_set)
x_email = vectorizer.transform([email_preprocessed])
prediction = clf.predict(x_email)[0]
probability = clf.predict_proba(x_email)[0]

print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
print(f"Confidence: {max(probability):.2%}")
```

## 📁 Project Structure

```
spam-classification/
│
├── spam_classifier.py          # Main script
├── spam_ham_dataset.csv        # Dataset (add your own)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── visualizations/             # Generated plots
│   ├── 01_data_distribution.png
│   ├── 02_text_length_distribution.png
│   ├── 03_word_clouds.png
│   ├── 04_confusion_matrix.png
│   └── 05_feature_importance.png
│
└── models/                     # Saved models (optional)
    ├── vectorizer.pkl
    └── classifier.pkl
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision (Spam) | ~96% |
| Recall (Spam) | ~95% |
| F1-Score (Spam) | ~95% |

*Note: Actual performance may vary based on dataset*

## 📈 Visualizations

### Data Distribution
Shows the balance between spam and ham emails in the dataset.

### Text Length Analysis
Compares the length distribution of spam vs. legitimate emails, revealing patterns.

### Word Clouds
Visual representation of the most frequent words in spam and ham emails.

### Confusion Matrix
Detailed breakdown of true positives, false positives, true negatives, and false negatives.

### Feature Importance
Top 20 words that contribute most to the classification decision.

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Scikit-learn** - Machine learning framework
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Word frequency visualization

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add real-time email classification API
- [ ] Support multi-language spam detection
- [ ] Deploy as web application using Flask/FastAPI
- [ ] Integrate with email clients (Gmail, Outlook)
- [ ] Add model versioning and experiment tracking (MLflow)
- [ ] Implement active learning for continuous improvement
- [ ] Add email header analysis for better detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**João Rafael Avansini Baião

- GitHub: [@JRBaiao](https://github.com/JRbaiao)
- LinkedIn: https://www.linkedin.com/in/jo%C3%A3o-rafael-a-bai%C3%A3o-466b16283/
- Email: joaorafaelavancinibaiao4@gmail.com

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data
- Inspired by various NLP and spam detection research
- Thanks to the open-source community

---

⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ and Python**
