## 다수결 투표 코드

> **1. 초기 Import 작업**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 전처리용 모듈
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# 훈련/검증용 분리 모듈
from sklearn.model_selection import train_test_split

# ensemble로 이용할 모델 import
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ensemble Voting 이용할 것 import
from sklearn.ensemble import VotingClassifier

# ROC 곡선 그리기
from sklearn.metrics import roc_curve, auc

# 최적화
from sklearn.model_selection import cross_validate, learning_curve, validation_curve, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

```

> **2. 데이터 가져오고 확인하기**

```python
# 데이터셋 가져오기
bank_df = pd.read_csv('ML/chapter3/UniversalBank.csv')
print(bank_df.head())
```

> **3. Data와 Target X, y로 분리**

```python
# Data와 Target X, y로 분리
X = bank_df.drop(['ID', 'ZIPCode', 'PersonalLoan'], axis=1)
print(X.head())

y = bank_df['PersonalLoan']
```

> **4. 데이터 전처리**

```python
# X 레이블 인코딩 (Education 가변수화)
# 특성 이름을 Education_Under 처럼 하기 위해서
X['Education'] = X['Education'].replace([1,2,3], ['Under', 'Grad', 'Prof'])
print(X['Education'].head())

X = pd.get_dummies(X[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']], columns=['Education'], drop_first=True)
print(X.head())

# y 레이블 인코딩
# 숫자형으로 되어 있기 때문에 변환 없음
print(y.head())
```

> **4.5 Train 세트와 Test 세트 분리**

```python
# X,y를 이용하여 트레이닝 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
```

> **5. 모델구축**

```python
# 모델 구축
# ensemble Voting에 들어갈 모델들 구축
logistic = LogisticRegression(solver='liblinear', penalty='l2', C=0.001, random_state=1)
tree = DecisionTreeClassifier(max_depth=None, criterion='entropy', random_state=1)
knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

# Voting에 들어갈 모델 튜플로 넣기
# 동시에 들어가게 만들기 위해서
voting_estimators = [('logistic', logistic), ('tree', tree), ('knn', knn)]

# Voting 만들기
voting = VotingClassifier(estimators = voting_estimators, voting='soft')
```

> **6. 모델검정**

```python
# 모델 검증할 때 쓰기 위한 값
clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'Majority voting']
all_clf = [logistic, tree, knn, voting]

# 반복문을 이용해서 검증하기
# 원래는 voting만 해주면 되는데, 얼마나 정확한지 알기 위해서
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                            X=X_train, y=y_train,
                            cv=10, scoring='roc_auc')

    print("ROC AUC : %0.3f (+/- %0.3f) [%s]" %(scores.mean(), scores.std(), label))
    
```

> **7. ROC 곡선 그리기**

```python
# ROC 그래프 그리기 
# 실제 값 중에 우리가 관심이 있는 것 잘맞춘 것 : TPR - 민감도
# 학습데이터 중에서 컴퓨터가 실제로 맞춘 정도 (맞춘 기준은 내가 설정한 특성이 맞았는지를)
# 예 ) T F T F T F 일 경우 나는 T가 얼마나 잘 맞췄는지 알고 싶다
# 컴퓨터 예측 T, T, T, T, T, T -> T는 컴퓨터가 엄청 잘 맞췄다 하지만 F는 전부 틀렸다... 그래서 1
# 실제 값 중에 우리가 관심이 없는 것 못맞춘 것 : FPR - 특이도
# 학습데이터 중에서 컴퓨터가 틀린 맞춘 정도 (맞춘 기준은 내가 설정하지 특성이 틀렸는지를)
# 컴퓨터 예측 T, T, T, T, T, T -> T는 컴퓨터가 엄청 잘 맞췄다 하지만 F는 전부 틀렸다... 그래서 1

# 예측 값 중에 잘맞춘 것 : ???
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label="%s (auc = %0.3f)" %(label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.xlim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")

plt.show()
```