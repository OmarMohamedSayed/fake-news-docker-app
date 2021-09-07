from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import json

from nltk.util import pr
app = Flask(__name__)
ps = PorterStemmer()
nltk.download('stopwords')
model = pickle.load(open('model2.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

def predict(text):
    review = text.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction

@app.route('/', methods=['GET'])
def home():
     return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)



# @app.route('/predict', methods=['GET','POST'])
# def predict_value():
#     text = request.args.get("text")
#     value = predict(text)
#     return jsonify(value=value)

if __name__ == "__main__":
    app.run()