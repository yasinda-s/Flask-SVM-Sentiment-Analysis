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
    prediction = svm_model.predict(text)
    prediction_proba = svm_model.predict_proba(text)
    return [prediction, prediction_proba]

def get_sentiment(prediction):
    if prediction == '1':
        return 'Positive'
    elif prediction == '0':
        return 'Neutral'
    else:
        return 'Negative'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['text']
        try:
            prediction, proba = predict(user_text)
            sentiment = get_sentiment(str(prediction[0]))
            return render_template('result.html', user_text=user_text, sentiment=sentiment, proba=proba)
        except:
            return 'Something went wrong'
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)