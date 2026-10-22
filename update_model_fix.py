import pickle
import sys
import os

# Determine active model path (matching app.py priority)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "model_improvement")
CAND_VEC_PATH = os.path.join(MODEL_DIR, "candidate_vectorizer.pkl")
CAND_MODEL_PATH = os.path.join(MODEL_DIR, "candidate_model.pkl")

if os.path.exists(CAND_VEC_PATH) and os.path.exists(CAND_MODEL_PATH):
    VECTOR_PATH = CAND_VEC_PATH
    MODEL_PATH = CAND_MODEL_PATH
else:
    VECTOR_PATH = 'best_vectorizer.pkl'
    MODEL_PATH = 'best_model.pkl'

# Load vectorizer and model
vectorizer = pickle.load(open(VECTOR_PATH, 'rb'))
model = pickle.load(open(MODEL_PATH, 'rb'))

# Example that is currently misclassified
sentence = "Everyone hates you, you should just give up on life."
# According to training data, bullying label is -1
label = -1

# Transform sentence
X = vectorizer.transform([sentence])

# Ensure the model supports partial_fit; if not, fallback to fit on combined data (not implemented here)
if hasattr(model, 'partial_fit'):
    # Perform a partial fit with the correct label
    model.partial_fit(X, [label], classes=model.classes_ if hasattr(model, 'classes_') else [-1, 0])
    # Save updated model
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    print(f'Model ({MODEL_PATH}) updated with example and saved.')
else:
    print('Model does not support partial_fit. Manual retraining required.', file=sys.stderr)

