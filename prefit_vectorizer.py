import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your CSV dataset
df = pd.read_csv("C:/Users/Admin/Downloads/cyberbullying_detection-main/cyberbullying_detection-main/indian_cyberbullying_dataset.csv")

# Optional: remove empty comments
df = df[df['comment_text'].notnull()]

# Load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Create and fit the vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True)
vectorizer.fit(df['comment_text'])

# Save the fitted vectorizer
with open("tfidfvectoizer_fitted.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("tfidfvectoizer_fitted.pkl has been created successfully!")
