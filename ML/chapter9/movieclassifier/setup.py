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