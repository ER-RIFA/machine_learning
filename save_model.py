import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from imblearn.over_sampling import RandomOverSampler
import joblib

print("Downloading NLTK required resources. This will only run once and may take a moment.")
try:
    nltk.data.find('corpora/stopwords.zip')
    nltk.data.find('tokenizers/punkt.zip')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

print("Loading dataset...")
try:
    df = pd.read_csv('trip_advisor_hotel_reviews.csv')
    df = df[['Review', 'Rating']]
except FileNotFoundError:
    print("Error: 'trip_advisor_hotel_reviews.csv' not found. Please place the file in the same directory.")
    exit()

def map_rating_to_sentiment(rating):
    if rating > 3:
        return 'positive'
    elif rating < 3:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['Rating'].apply(map_rating_to_sentiment)
df = df.dropna()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered_tokens)

df['cleaned_review'] = df['Review'].apply(preprocess_text)

X = df['cleaned_review']
y = df['sentiment']

print("Balancing dataset...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)
X_resampled = X_resampled.flatten()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print("Training model...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
model = LinearSVC()
model.fit(X_train_vectorized, y_train)

print("Saving model files...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully. You can now run the API.")
