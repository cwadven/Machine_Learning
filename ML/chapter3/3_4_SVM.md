## 서포트 벡터 머신을 사용항 최대 마진 분류

> **서포트 벡터 머신**

<ul>
<li>강력하고 널리 사용되는 학습 알고리즘.</li>
<li>퍼셉트론의 확장</li>
<li>마진을 최대화하는 것</li>
</ul>

마진은 클래스를 구분하는 초평면(결정 경계)과 이 초평면에 가장 가까운 훈련 샘플 사이의 거리로 정의한다.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/SVM.PNG" alt="drawing" width="600"/><br><br>

결정경계를 나누는데, 그 중에서 최적화된 직선을 구하기 위해서 가장 바깥쪽에 있는 데이터(서포트 벡터)가 서로 상대방과 마주보는 평행한 직선을 구한다.

- 용어
    - 마진
    - 초평면 == 결정 경계
    - 서포트 벡터 (가장 바깥쪽 데이터)

순서

서포트 백터를 통해 2개의 직선을 구한다

수평인 직선과 직선 둘 사이의 거리 가장 최대화 되는 거리는 직각일 경우 `|| ||` 이걸 `노름`이라고 한다.

---

### 수학적 지식 : 선형대수 (원점과 점 사이 거리 : 노름)

> **거리 : 피타고라스의 정리**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/pitagoras.PNG" alt="drawing" width="600"/><br><br>


> **좌표의 거리 구하기**

- 유클리드 거리

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/coord.PNG" alt="drawing" width="600"/><br><br>

- 유클리드 거리 예)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/coord1.PNG" alt="drawing" width="600"/><br><br>

> **노름**

- 원점에서 점에 이르는 거리 (점 1개)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/norm.PNG" alt="drawing" width="450"/><br><br>

- 유클리디안 노름 (점 1개, 만약 다차원이면 )

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/norm3.PNG" alt="drawing" width="450"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/ucle.PNG" alt="drawing" width="450"/><br><br>

---

### 최대마진 구하기

> **초평선 구하는 방법**

---

> **최대 마진**

큰 마진의 결정 경계를 원하는 이유는 일반화 오차가 낮아지는 경향이 있기 때문이다.

반면에 작은 마진의 모델은 과대적합되기 쉽다.

~~~
과대적합 : 되면 좋지 못한것...
~~~

---

### SVM 이용한 코드

```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

# 얼마나 예측 잘 했는지 확인 하기 위해서 사용
from sklearn.metrics import accuracy_score

# csv 파일 가져오기
dataset = pd.read_csv("ML/chapter3/accidentsnn.csv")

# X랑 y 분리 시키기
X = dataset.drop(["MAX_SEV_IR"], axis=1)
y = dataset["MAX_SEV_IR"]

# 분리 시켜줄 것 생각해 보기
sc = StandardScaler()

# Standard 적용한 값 슬라이싱을 통해 재할당 하기!
# X.loc[[행], [열]] : 이름으로 접근 / X.iloc[[행], [열]] : index로 접근
X.iloc[:, [3]] = sc.fit_transform(X.iloc[:, [3]])


X["ALCHL_I"] = X["ALCHL_I"].replace([1,2], ['Yes', 'No'])
X["SUR_COND"] = X["SUR_COND"].replace([1,2,3,9], ['dry', 'wet', 'snow', 'non'])


# 수치화된 데이터를 가변수화
# 가변수화를 하는 이유는 만약 1:월, 2:화, 3:수 일 경우 1 + 2 = 3 이기에 수요일과 관계가 생길 수 있기 때문
# get_dummies(dataframe, dataframe에 가변수화 할 열, drop_first=True)
# drop_first : 가변수화할 때 이상한 값이 나오게 됩니다! 그거 막기 위함
X = pd.get_dummies(X[["ALCHL_I", "PROFIL_I_R", "VEH_INVL", "SUR_COND"]],
columns=["ALCHL_I", "SUR_COND"], drop_first=True)


# 전처리를 했으니 학습용 데이터와 검증용 데이터 분류
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# SVM 알고리즘 사용
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train, y_train)

# 모델 검증
y_pred = svm.predict(X_test)

print('잘못 분류된 샘플 개수 : %d' % (y_test != y_pred).sum())
print('정확도: %.2f' % accuracy_score(y_test, y_pred))
```