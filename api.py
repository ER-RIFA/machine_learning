import joblib
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import ssl
import nltk
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


app = Flask(__name__)

try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("All NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please check your internet connection and try again.")

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered_tokens)

try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    print("Please run 'save_model.py' to generate them.")

    try:
        print("Training model from scratch...")
        df = pd.read_csv('trip_advisor_hotel_reviews.csv')
        df = df[['Review', 'Rating']]
        
        def map_rating_to_sentiment(rating):
            if rating > 3: return 'positive'
            elif rating < 3: return 'negative'
            else: return 'neutral'

        df['sentiment'] = df['Rating'].apply(map_rating_to_sentiment)
        df = df.dropna()
        df['cleaned_review'] = df['Review'].apply(preprocess_text)

        X = df['cleaned_review']
        y = df['sentiment']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vectorized, y_train)

        model = LinearSVC()
        model.fit(X_train_resampled, y_train_resampled)
        
        print("Model training complete.")
        
    except Exception as e:
        print(f"Failed to train model: {e}")
        exit()

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    """
    API endpoint to predict the sentiment of a given review text.
    It expects a JSON payload with a 'review' key.
    """
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Please provide a "review" in the request body.'}), 400

    review_text = data['review']
    
    cleaned_review = preprocess_text(review_text)
    
    vectorized_text = vectorizer.transform([cleaned_review])
    
    predicted_sentiment = model.predict(vectorized_text)[0]
    
    return jsonify({
        'review': review_text,
        'predicted_sentiment': predicted_sentiment
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
