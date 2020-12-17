## 파이썬으로 퍼셉트론 학습 알고리즘 구현

**구축 방법**

1. 가상환경 생성 및 실행
[머신러닝을 위한 파이썬 및 구축 방법](https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/1_5_machine_python.md "머신러닝을 위한 파이썬 및 구축 방법")
<br>

2. numpy 모듈 설치
~~~
pip install numpy
pip install matplotlib
pip install pandas
~~~

3. 퍼셉트론 알고리즘 코드 구현

```python
import numpy as np

class Perceptron(object):
    """퍼셉트론 분류기

    매개변수
    -----------------
    eta : float
        학습률(0.0과 1.0 사이)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드
    
    속성
    -----------------
    w_ : 1d-array (1차원 배열)
        학습된 가중치
    errors_ : list
        에포크 epoch (학습한 횟수)마다 누적된 분류 오류
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """훈련 데이터 학습

        매개변수
        -----------------
        X : {array-like}, shape = [n_samples, n_features]
            n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]

        반환값
        -----------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환한다"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

4. 붓꽃 데이터 이용하기 위해 가져오기 (잘 가져와 지는지 확인)

```python
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# 마지막 5개 가져온 것 보기
df.tail()
```

5. 붓꽃 데이터 시각화 하여 퍼셉트론 알고리즘에 쓸 수 있는지 확인

~~~
퍼셉트론은 두 클래스가 선형적으로 구분되고 학습률이 충분히 작을 때만 수렴을 보장한다.

두 클래스를 선형 결정 경계로 나눌 수 없다면 훈련 데이터셋을 반복할 최대 횟수(epoch)를 지정하고 분류 허용 오차를 지정해야한다.

그렇지 않으면 계속 가중치를 업데이트 한다!
~~~

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/linaer.PNG" alt="drawing" width="600"/><br>

```python
import matplotlib.pyplot as plt
import numpy as np

# 붓꽃 데이터의 setosa와 versicolor를 선택
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 꽃받침 길이와 꽃잎 길이를 추출합니다.
X = df.iloc[0:100, [0,2]].values

# 산점도를 그린다. (시각적으로 보기 위해서)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()
```

6. 붓꽃 데이터로 만든 퍼셉트론 알고리즘으로 예측하기