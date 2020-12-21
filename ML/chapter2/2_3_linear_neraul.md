## 적응형 선형 뉴런과 학습의 수렴

> **선형 뉴런 (ADALINE)**<br>
아달린은 연속 함수로 비용 함수를 정의하고 최소화 하는 핵심 개념을 보여준다.<br>
`로지스틱 회귀`, `서포트 백터 머신` 같은 분류를 위한 고급 머신러닝 모델과 회귀 모델을 이해하는데 도움이 될 것이다.

> **퍼셉트론과 아달린의 차이점**<br>
퍼셉트론 임계함수를 통해 컴퓨터가 예측하는 것을 -1 혹은 1로 정의<br>
아달린 z를 바로 실제 값과 바로 - 함.<br><br>
`퍼셉트론`은 n(y-y)x를 하여 △w를 구해 이용하는데,<br>
`아달린`은 n(y-y)x을 하는데 한 사이클이 끝나고 그 모든 x특성의 △w를 더한다.
그리고 그 △w를 이전의 w와 더해서 가중치를 구한다.

> **경사 하강법**<br>
<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/gradient.PNG" alt="drawing" width="150"/><br>
위의 식에서 1/2는 그래디언트를 간소하게 만들려고 편의상 추가한 것이다.
전역 최솟값에 도달할 때까지 언덕을 내려오는 것으로 묘사.
<br>

**경사하강법**<br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/gradient2.PNG" alt="drawing" width="600"/><br>

---

### 아달린 순차적 방법

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s1.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s2.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s3.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s4.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s5.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s6.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s7.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s8.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter2/img/s9.PNG" alt="drawing" width="600"/><br><br>

**`끝난 후 예측은`** <br><br>

최적의 가중치 w들을 구했으면 예측하고 싶은 특성을 해당 w에 넣어서 z를 구하고, z를 임계함수 안에 넣어서 분류 시킨다.

---

### 파이썬으로 아달린 구현

#### 1. 아달린 클래스 구현

```python
class AdalineGD(object):
    """적응형 선형 뉴런 분류기

    매개변수
    -----------------
    eta : float
        학습률 (0.0과 1.0사이)
    
    n_iter : int
        훈련 데이터셋 반복 횟수

    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------------
    w_ : 1d-array
        학습된 가중치
    
    cost_ : list
        에포크마다 누적된 비용 함수의 제곱합
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
            타깃값
        
        반환값
        -----------------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """선형 활성화 계산"""
        return X

    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
```