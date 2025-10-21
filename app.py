from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.fit_transform([user_input])  # keeping exactly as your original
        prediction = model.predict(transformed_input)[0]
    
    return render_template('index.html', prediction=prediction)

# 🟢 Added these routes (nothing else changed)
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)



print("loser" in vectorizer.vocabulary_)
