import pandas as pd
import re
import nltk
import ssl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('all', quiet=True)
    print("All NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please check your internet connection and try again.")
    exit()

try:
    df = pd.read_csv('trip_advisor_hotel_reviews.csv')
    
    df = df[['Review', 'Rating']]

    print("Dataset loaded successfully.")
    print(df.head())

except FileNotFoundError:
    print("Error: 'trip_advisor_hotel_reviews.csv' not found.")
    print("Please ensure the filename is exactly 'trip_advisor_hotel_reviews.csv' and that it is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("\n--- Training SVM Model ---")
model = LinearSVC()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Try it Yourself! ---")
while True:
    new_review = input("Enter a new review (or type 'quit' to exit): \n")
    if new_review.lower() == 'quit':
        break
    
    cleaned_new_review = preprocess_text(new_review)
    
    new_review_vectorized = vectorizer.transform([cleaned_new_review])
    
    predicted_sentiment = model.predict(new_review_vectorized)[0]
    
    print(f"Predicted Sentiment: {predicted_sentiment}\n")
