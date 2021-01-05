## 8장 내용중 학습된 사잇킷런 추정기 저장

### I. 학습한 모델 바이트 형태로 빼오기

1. movie_data.csv 정보를 가지고 리뷰한 것을 문자열를 한번 가공한다.

2. 그 후 중요한 것들만 가져오도록 가져오는 HashingVectorizer 라는 틀을 sklearn 모듈을 이용해서 가져온다.

3. 1번에서 가공한 문자열을 HashingVectorizer 틀에 넣어서 X_train을 만들다.

4. 가공한 것을 이용해서 모델을 학습한다.

5. 학습한 모델을 pickle로 dump 시켜 모델을 내보낸다.<br>
(stop은 불용어 즉 running 일경우 run으로 변환 같은 것을 해주는 것)

```python
import os
import gzip


if not os.path.isfile('movie_data.csv'):
    if not os.path.isfile('movie_data.csv.gz'):
        print('Please place a copy of the movie_data.csv.gz'
              'in this directory. You can obtain it by'
              'a) executing the code in the beginning of this'
              'notebook or b) by downloading it from GitHub:'
              'https://github.com/rasbt/python-machine-learning-'
              'book-2nd-edition/blob/master/code/ch08/movie_data.csv.gz')
    else:
        in_f = gzip.open('movie_data.csv.gz', 'rb')
        out_f = open('movie_data.csv', 'wb')
        out_f.write(in_f.read())
        in_f.close()
        out_f.close()

import numpy as np
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# `stop` 객체를 앞에서 정의했지만 이전 코드를 실행하지 않고
# 편의상 여기에서부터 코드를 실행하기 위해 다시 만듭니다.
stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # 헤더 넘기기
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path='movie_data.csv'))

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        pass
    return docs, y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

doc_stream = stream_docs(path='movie_data.csv')

import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('정확도: %.3f' % clf.score(X_test, y_test))

import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')

if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)

pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
```

---

### II. 빼온 학습 데이터 읽어서 사용해보기

1. **사용하기 위해서 가공 시키기**

stop 이라는 것을 이용해서 running 같은 단어가 들어오면 run으로 만들어주는 기능으로 여러 단어 가공 시키기 위한 코드 작성

- 전처리 과정

**`vectorizer.py 파일 생성하여 그 안에 작성`**

```python
from sklearn.feature_extraction.text import HashingVectorizer

import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                        n_features=2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)
```

2. 가공 처리하는 것 까지 한 뒤, 그 해당 가공을 통해서 입력 값을 만들고 그것은 훈련시킨 모델에 대입!

```python
import pickle
import re
import os
from vectorizer import vect

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

import numpy as np

label = {0: '음성', 1: '양성'}

example = ['I love this movie']

X = vect.transform(example)

print('예측 : %s\n 확률: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
```


---

### III. Flask를 이용해서 웹 애플리케이션으로 가져오면서 계속 학습 시키기

- movieclassifier 폴더에 해당 자료 있음

#### 필요한 모듈

~~~
pip install flask
pip install wtforms
pip install sklearn
pip install numpy
pip install nltk
pip install pyprind
~~~

#### 실행

~~~
$cd 1st_flask_app_1
$python app.py
~~~