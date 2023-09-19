from flask import Flask, render_template, url_for, request, redirect
import pickle

app = Flask(__name__)

def load_model():
    with open('SVMModel', 'rb') as f:
        svm_model, vectorizer = pickle.load(f)
    return svm_model, vectorizer

def predict(text):
    svm_model, vectorizer = load_model()
    text = vectorizer.transform([text])
    return svm_model.predict(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['text']
        prediction = predict(user_text)
        return str(prediction[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)