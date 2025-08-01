# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def load_dataset():
    """Load the dataset from CSV file"""
    try:
        # Load dataset with specified columns
        df = pd.read_csv("C:/Users/Raja Abdullah/Desktop/Internship Projects/SMS Spam Detection System in Python/spam.csv", encoding='latin-1')
        df = df[['v1', 'v2']]  # Select only relevant columns
        df.columns = ['label', 'message']  # Rename columns for clarity
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None



def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)



def prepare_data(df):
    """Prepare data for modeling"""
    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split into features and target
    X = df['processed_message']
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test



def vectorize_text(X_train, X_test):
    """Convert text to numerical features using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer



def train_model(X_train, y_train):
    """Train a Naive Bayes classification model"""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return y_pred, accuracy


def show_predictions(model, vectorizer, X_test, y_test, accuracy, n=5):
    """Display sample predictions with accuracy"""
    test_messages = X_test.sample(n, random_state=42)
    actual_labels = y_test.loc[test_messages.index]
    
    processed_messages = test_messages
    vectorized_messages = vectorizer.transform(processed_messages)
    predictions = model.predict(vectorized_messages)
    
    print("\nSample Predictions:")
    for i, (msg, actual, pred) in enumerate(zip(test_messages, actual_labels, predictions)):
        print(f"\nExample {i+1}:")
        print(f"Message: {msg}")
        print(f"Actual: {'Spam' if actual == 1 else 'Ham'}")
        print(f"Predicted: {'Spam' if pred == 1 else 'Ham'}")
    print(f"\nOverall Test Accuracy: {accuracy:.2%}")



def visualize_spam_words(df):

    # Bar plot of top 20 spam words
    spam_word_counts = pd.Series(' '.join(df[df['label'] == 1]['processed_message']).split()).value_counts()[:20]
    
    plt.figure(figsize=(10, 6))
    spam_word_counts.plot(kind='bar')
    plt.title('Top 20 Most Common Words in Spam Messages')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def save_model(model, vectorizer, filename='spam_detector.joblib'):
    """Save the trained model and vectorizer"""
    joblib.dump({'model': model, 'vectorizer': vectorizer}, filename)
    print(f"\nModel saved as {filename}")



def interactive_predict(model, vectorizer):
    """Interactive CLI for spam detection"""
    print("\nInteractive Spam Detection (type 'quit' to exit)")
    while True:
        message = input("\nEnter a message to check: ")
        if message.lower() == 'quit':
            break
        
        processed = preprocess_text(message)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        
        print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")

def main():
    # Load dataset
    df = load_dataset()
    
    if df is None:
        return
    
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Vectorize text using TF-IDF
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # Train Naive Bayes model
    model = train_model(X_train_vec, y_train)
    
    # Evaluate model
    y_pred, accuracy = evaluate_model(model, X_test_vec, y_test)
    
    # Show sample predictions
    show_predictions(model, vectorizer, X_test, y_test, accuracy, n=5)
    
    # Visualize spam words
    visualize_spam_words(df)
    
    # Save model
    save_model(model, vectorizer)
    
    # Interactive prediction
    interactive_predict(model, vectorizer)

if __name__ == "__main__":
    main()
