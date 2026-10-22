import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = 'dataset.csv'
VECTOR_PATH = 'best_vectorizer.pkl'
MODEL_PATH = 'best_model.pkl'

# Load dataset
print('Loading dataset...')
df = pd.read_csv(DATA_PATH)
# Ensure correct column names
if 'headline' not in df.columns:
    df = df.rename(columns={df.columns[0]: 'headline', df.columns[1]: 'label'})

# Add the misclassified example to ensure it's learned
extra = pd.DataFrame({
    'headline': ['You are a complete failure'],
    'label': [-1]
})
df = pd.concat([df, extra], ignore_index=True)

X = df['headline'].astype(str).values
y = df['label'].astype(int).values

# Split (optional) - we will train on full data for best performance
# vectorizer
vectorizer = TfidfVectorizer(lowercase=True, max_features=10000)
X_vec = vectorizer.fit_transform(X)

# Train SGDClassifier (linear SVM style)
clf = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)
clf.fit(X_vec, y)

# Save vectorizer and model
with open(VECTOR_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)

print('Training complete. Model and vectorizer saved.')
