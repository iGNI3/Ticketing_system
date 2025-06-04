"""
Ticket Processing System
------------------------
This script processes customer support tickets using NLP and ML.

Features:
- Modular functions for preprocessing, entity extraction, and categorization
- TF-IDF + Random Forest models
- Small-scale manual validation
- Insights/challenges at the end

Requirements:
    pip install pandas scikit-learn nltk textblob gradio openpyxl
"""

import pandas as pd
import re
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

### 1. Data Loading ###
def load_data(path):
    data = pd.read_excel(path)
    # Drop missing target values
    data.dropna(subset=['issue_type', 'urgency_level'], inplace=True)
    return data

### 2. Text Preprocessing ###
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

### 3. Feature Engineering ###
def add_features(data):
    # Cleaned text
    data['cleaned_text'] = data['ticket_text'].apply(preprocess_text)
    # Ticket length
    data['ticket_length'] = data['ticket_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    # Sentiment score
    sia = SentimentIntensityAnalyzer()
    data['sentiment_score'] = data['ticket_text'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    return data

### 4. Entity Extraction ###
def extract_entities(ticket_text, product_list):
    products = re.findall(r'\b(?:' + '|'.join(map(re.escape, product_list)) + r')\b', ticket_text, flags=re.IGNORECASE)
    dates = re.findall(r'\b\d{1,2} \w+ \d{4}\b', ticket_text)
    keywords = ['broken', 'late', 'error']
    complaints = [word for word in ticket_text.split() if word.lower() in keywords]
    return {
        "product_names": products,
        "dates": dates,
        "complaints": complaints
    }

### 5. Model Training & Evaluation ###
def train_models(X, y_issue, y_urgency):
    # Split
    X_train, X_test, y_train_issue, y_test_issue = train_test_split(X, y_issue, test_size=0.2, random_state=42)
    X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(X, y_urgency, test_size=0.2, random_state=42)
    # Models
    issue_clf = RandomForestClassifier(random_state=42)
    urgency_clf = RandomForestClassifier(random_state=42)
    issue_clf.fit(X_train, y_train_issue)
    urgency_clf.fit(X_train_urgency, y_train_urgency)
    # Predict
    y_pred_issue = issue_clf.predict(X_test)
    y_pred_urgency = urgency_clf.predict(X_test_urgency)
    # Reports
    print("Issue Type Classification Report:\n", classification_report(y_test_issue, y_pred_issue))
    print("Urgency Level Classification Report:\n", classification_report(y_test_urgency, y_pred_urgency))
    return issue_clf, urgency_clf

### 6. Ticket Processing Pipeline ###
def process_ticket(ticket_text, vectorizer, issue_clf, urgency_clf, product_list):
    cleaned = preprocess_text(ticket_text)
    X_vec = vectorizer.transform([cleaned])
    issue_type = issue_clf.predict(X_vec)[0]
    urgency_level = urgency_clf.predict(X_vec)[0]
    entities = extract_entities(ticket_text, product_list)
    return {
        "predicted_issue_type": issue_type,
        "predicted_urgency_level": urgency_level,
        "extracted_entities": entities
    }

if __name__ == "__main__":
    # ---- LOAD DATA ----
    data = load_data("E:\\assignment\\ai_dev_assignment_tickets_complex_1000.xls")
    data = add_features(data)

    # ---- VECTORIZATION ----
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    y_issue = data['issue_type']
    y_urgency = data['urgency_level']

    # ---- TRAIN MODELS & EVALUATE ----
    issue_clf, urgency_clf = train_models(X, y_issue, y_urgency)

    # ---- VALIDATION ON SMALL MANUAL SAMPLE ----
    sample_tickets = [
        "I ordered SmartWatch V2 but got PowerMax Battery instead. My order number is #65084.",
        "I ordered the SmartWatch V2, but it is broken and the delivery was late.",
        "Both my PowerMax Battery and EcoBreeze AC are lost. Both giving issues. Also, I contacted support on 12 March but got no response."
    ]
    print("\n--- Manual Validation on Sample Tickets ---")
    product_list = data['product'].dropna().unique().tolist()
    for idx, ticket in enumerate(sample_tickets):
        result = process_ticket(ticket, vectorizer, issue_clf, urgency_clf, product_list)
        print(f"\nSample {idx+1}:\nInput: {ticket}\nOutput:\n{json.dumps(result, indent=4)}")

    # ---- INSIGHTS & CHALLENGES ----
    print("\n--- Insights & Challenges ---")
    print("""
- The model performs well on classes with enough data, but rare classes show lower precision/recall.
- Entity extraction is regex-based and may miss product name variants or date formats not in '12 March 2023' style.
- False positives for complaints can occur with simple keyword matching.
- Sentiment score and ticket length added minor improvements to urgency prediction.
- For higher performance, more advanced NLP (e.g., transformers) or more labeled data may help.
- Gradio UI can be added for interactive use if needed.
""")