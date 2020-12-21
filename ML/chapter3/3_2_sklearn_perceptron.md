## 사이킷런 첫걸음 : 퍼셋트론 훈련

**구축 방법**

1. 가상환경 생성 및 실행
[머신러닝을 위한 파이썬 및 구축 방법](https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/1_5_machine_python.md "머신러닝을 위한 파이썬 및 구축 방법")
<br>

2. numpy 모듈 설치
~~~
pip install numpy
pip install matplotlib
pip install pandas
pip install sklearn
~~~

3. sklearn을 이용하여 퍼셉트론구현

```python
# 데이터 가져오기
from sklearn import datasets
# x의 입력값을 0~1 사이로
from sklearn.preprocessing import StandardScaler
# 데이터를 쉽게 train이랑 test로 분리
from sklearn.model_selection import train_test_split
# 퍼셉트론 알고리즘 사용
from sklearn.linear_model import Perceptron

import numpy as np

# 얼마나 예측 잘 했는지 확인 하기 위해서 사용
from sklearn.metrics import accuracy_score


# 아이리스 자료 가져오기
iris = datasets.load_iris()
# 가져온 자료에서 필요한 특성만 빼기
X = iris.data[:, [2,3]]
# 자료의 라벨 즉 무슨 꽃인지를 넣어 놓은 것을 이용하기
y = iris.target

# 각각의 변수에 train_test_split 메소드를 이용하여 X와 y의 값을
# 훈련과 테스트용을 분리 시킨다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# x의 특성 값을 0~1 사이로 초기화
sc = StandardScaler()
# 형태 틀 만들기
sc.fit(X_train)
# 틀에 넣어서 0~1로 만들기
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 퍼셉트론 알고리즘 객체 생성
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
# 학습시키기 (변환한0~1값을가진특성, 라벨) / 배열 : (2차원, 1차원)
ppn.fit(X_train_std, y_train)

# 학습된 모델로 X_test_std 특성을 넣어서 y라벨 예측하기
y_pred = ppn.predict(X_test_std)

# 정확도 비교하기
# 정확도 = "1 - 오차"
# 오차 = 틀린 개수 / 전체 개수 = 오차
# 틀린 개수 : y_test의 값과 y_pred의 각 인덱스가 다른 것들의 개수를 샌다
# numpy -> (y_test != y_pred).sum()
# 혹은 sklearn에서 제공하는 정확도를 바로 구하는 함수 사용
print('정확도: %.2f' % accuracy_score(y_test, y_pred))
```