## 결정 트리 학습

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/decision_tree.PNG" alt="drawing" width="600"/><br><br>

~~~
분류나무 (분류)
회귀나무 (예측)
~~~

<h5>결정 트리는 수치 데이터 입력 X를 정규화 시켜줄 필요가 없다</h5>

---

### 결정 트리 사용 알고리즘

`CART 알고리즘 = 지니 지수 (Gini index) 기준`<br>
- 지니 지수 : 불확실성(지니) 낮아지는게 좋음<br>
이지분리 / 학습데이터 -> 검증

`C4.5 알고리즘 = 엔트로피 지수 (Entropy index) 기준 / IG`<br>
다지분리 / 학습 데이터

`CHAID 알고리즘 = 카이제곱 통계량(Chi-Square statistic) 기준`<br>
다지분리

---

### CART 사용하는 방법 (Gini : 불확실성)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/gini.PNG" alt="drawing" width="150"/><br><br>

CART는 2번 추출한다! 그래서 제곱

G(A) = 0.5 (두 집단이 동일 할 때, 반반)

지니지수 : 최대치 0.5

그룹의 class가 동일하면 1 - 불확실성 = 떨어진다.<br>
그룹의 class가 다르면 1 - 불확실성 = 올라간다.

**예 ) 충성고객과 탈퇴고객을 구분하는 규칙을 생성하자!**<br>
총 10명의 고객을 대상으로 `성별`과 `결혼유뮤` 중 어느 변수가 더 분류를 잘하는 변수인지 찾고, 분류규칙을 찾고자함

> **성별**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART1.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART2.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART3.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART4.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART5.PNG" alt="drawing" width="600"/><br><br>

> **결혼유무**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART6.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART7.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART8.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART9.PNG" alt="drawing" width="600"/><br><br>

> **결과**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/CART10.PNG" alt="drawing" width="600"/><br><br>

둘을 비교하여 gini 지수가 더 작은 것을 통해 좋은 변수를 알 수 있다!

---

### C4.5 사용하는 방법 (정보 이득 최대화 : entropy 엔트로피)

엔트로피를 이용해서 Information Gain(IG)을 획득!

~~~
정보이론(엔트로피) -> 정보가 얼마나 많은지... log 2 로 계산

log 2 (8) = 3

분수가 나온다

로그는 무조건 -가 나온다!<br>
(log 2 (1/2) = -1 이기 때문에)
~~~

---

### 엔트로피 계산 방법

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro1.PNG" alt="drawing" width="350"/><br><br>

두번을 곱해주는데, 한번은 log 2를 이용해서 곱해준다!

`-` 하기 귀찮으니 앞에 `-` 를 붙이는 것이다.

**그래서**

IG : 정보이익 = 정보의 가치! 높으면 좋다<br>
- IG = E(before:상위) - E(after:현재)

전의 엔트로피에서 후의 엔트로피 빼기 한 것이다.

전의 불확실성 - 후의 불확실성 = IG (그래서 값이 높아지는 것이 좋다)

**예)**

> **성별**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro2.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro3.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro4.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro5.PNG" alt="drawing" width="600"/><br><br>

> **결혼유무**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro6.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro7.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro8.PNG" alt="drawing" width="600"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/entro9.PNG" alt="drawing" width="600"/><br><br>

> **이득률 Information gain ratio**

~~~
가지가 2개일 경우는 상관이 없는데, C4.5 같이 다치가지를 가지는 경우!
IG가 적은 가지를 가지는 것과 비교하여 큰 값이 나온다!
이런 경우를 대비하여 패널티를 주기 위한 방법이다.
~~~

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/IGR.PNG" alt="drawing" width="350"/><br><br>

**조금 전의 예)**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/IGR2.PNG" alt="drawing" width="400"/><br><br>

---

### 의사결정 나무 과정
1. 나무모델 생성
    - CART, C4.5, CHAID 중 무엇을 선택할 지 고른다.

2. 과적합 문제 해결
    가지치기를 얼마나 할 것인지 알기 위해서<br>
    - 과소적합 : 훈련, 검증 정확도 모두 낮음 (학습 별로 안함) / 학습곡선 기준으로 확인
        - 편향이 높아야 한다
    - 과대적합 : 훈련 데이터에 비해 모델이 너무 복잡할 경우 (학습 데이터가 너무 많은 경우) / 검증곡선 기준으로 자른다
        - 분산이 적어야 한다

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/extend.PNG" alt="drawing" width="400"/><br><br>

3. 검증
4. 해석 및 예측

---

## 결정 트리 코드 (GINI 이용)

```python
from sklearn.tree import DecisionTreeClass

# 결정 트리 사용
tree = DecisionTreeClass(criterion='gini',
                        max_depth=4, random_state=1)

tree.fit(X_train, y_train)

# 결정 경계 보기 위한 값 설정
X_combined = np.vstack((X_train, X_test))
y_combined = np.vstack((y_train, y_test))

# 결정 경계 만들기 import 따로 필요 plot_decision_regions
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))

# 표 보이게 하기
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```