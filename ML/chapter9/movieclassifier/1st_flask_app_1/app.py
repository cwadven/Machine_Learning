from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

from vectorizer import vect

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

# 입력값 vect로 만들고 예측해서 결과 값 도출하기
def classifier(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

# 모델 학습 추가하기
def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

# DB에 넣은 내용 추가
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        # 결과 보여주기
        y, proba = classifier(review)
        return render_template('results.html', content=review, prediction=y, probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative' : 0, 'positive' : 1}
    y = inv_label[prediction]

    if feedback == '틀림':
        y = int(not(y))
    
    # 추가적으로 학습시키기
    train(review, y)

    sqlite_entry(db, review, y)

    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)