## 범주형 데이터 다루기

### 순서가 있는 특성과 순서가 없는 특성

~~~
범주형 데이터에 관해 이야기 할 때 순서가 있는 것과 없는 것을 구분해야 한다.
순서가 있는 특성은 정렬하거나 차례대로 놓을 수 있는 범주형 특성으로 생각할 수 있다.

예를 들어 티셔츠 사이즈는 XL > L > M 으로 순서를 정할 수 있으므로 순서가 있는 특성이다.

반대로 순서가 없는 특성은 차례를 부여할 수 없다.

예로 티셔츠 컬러는 순서가 없는 특성이다.
~~~

--------

#### 1. 예제 데이터셋 만들기 (값을 df으로 변환)

```python
import pandas as pd

# 해당 리스트를 이용하여 데이터프레임의 각 행을 만든다.
df = pd.DataFrame([
    ['grean', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1'],
])

# 행에 해당하는 각 열을 추가한다.
df.columns = ['color', 'size', 'price', 'classlabel']

# df 출력
print(df)

'''
출력 결과
    color    size    price   classlabel
0   grean     M       10.1     class1
1   red       L       13.5     class2
2   blue      XL      15.3     class1
'''
```

---

#### 2. 순서 특성 매핑

여기서는 특성 간의 산술적인 차이를 이미 알고 있다고 가정 한다.

`XL = L + 1 = M + 2`

```python
size_mapping = {
    'XL' : 3,
    'L' : 2,
    'M' : 1
}

# map이라는 메서드를 이용하여 size_mapping의 딕셔너리 값에 맞게 각각 매핑 시킨다
df['size'] = df['size'].map(size_mapping)

print(df)

'''
출력 결과
    color    size    price   classlabel
0   grean     1       10.1     class1
1   red       2       13.5     class2
2   blue      3       15.3     class1
'''

# 다시 원래대로 돌아가고 싶을 경우
inv_size_mapping = {v: k for k, v in size_mapping.items()}

print(inv_size_mapping)

'''
{ 3:'XL', 2:'L', 3:'M' }
'''

# map이라는 메서드를 이용하여 매핑
df['size']= df['size'].map(inv_size_mapping)
```

---

#### 3. 클래스 레이블 인코딩

클래스 레이블이 정수로 인코딩되었을 것을 기대하여 글자를 숫자화

> **(방법1) 직접 인코딩 시켜주기**
순서 특성을 매핑한 것과 비슷한 방식을 사용

~~~
클래스 레이블은 순서가 없다는 것을 기억!!

특정 문자열 레이블에 할당하는 정수는 의미 없는 정수이다!

그냥 class1은 0이고 class2는 1이다 라고 숫자로 나타내는 것 뿐!!
~~~

```python
import numpy as np

# enumerate라는 함수를 이용하여 클래스라벨의 값에 숫자를 넣어줘서 라벨링 시켜주는 것이다.
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}

'''
{ 'class1':0, 'class2':1 }
'''

df['classlabel'] = df['classlabel'].map(classlabel)

print(df)

'''
출력 결과
    color    size    price   classlabel
0   grean     1       10.1       0
1   red       2       13.5       1
2   blue      3       15.3       0
'''

# 다시 원래대로 돌리고 싶을 경우
inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

print(df)

'''
출력 결과
    color    size    price   classlabel
0   grean     1       10.1     class1
1   red       2       13.5     class2
2   blue      3       15.3     class1
'''
```

> **(방법2) 사이킷런에 구현된 LabelEncoder 클래스 사용**

```python
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

# LabelEncoder를 이용하여 레이블 인코딩
df['classlabel'] = class_le.fit_transform(df['classlabel'].values)

'''
출력 결과
    color    size    price   classlabel
0   grean     1       10.1       0
1   red       2       13.5       1
2   blue      3       15.3       0
'''

# LabelEncoder 이용하여 다시 원상태로 고치기
df['classlabel'] = class_le.invers_transfor(y)

'''
출력 결과
    color    size    price   classlabel
0   grean     1       10.1     class1
1   red       2       13.5     class2
2   blue      3       15.3     class1
'''
```

---

#### 4. 순서가 없는 특성에 원-핫 인코딩 적용

순서가 없는 특성도 LabelEncoder를 이용하여 간편하게 문자열 레이블 정수로 인코딩 할 수 있다.

```python
X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()

X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

'''
출력 결과
array([
    [1, 1, 10.1],
    [2, 2, 13.5],
    [0, 3, 15.3],
], dtype=object)
'''
```

**원-핫 인코딩을 하는 이유**<br>
컬러 값에 어떤 순서가 없지만 학습 알고리즘이 green은 blue보다 크고 red는 green 보다 크다고 가정할 수 있다!

이 가정이 옳지 않지만 알고리즘이 의미 있는 결과를 만들 수 있다.

하지만 이 결과는 최선이 아닐 것이다.

이 문제를 해결하기 위해서 `원-핫 인코딩` 기법을 이용한다.

> **원-핫 인코딩**
순서 없는 특성에 들어 있는 고유한 값마다 새로운 더미(dummy) 특성을 만드는 것!

```python
from sklearn.preprocessing import OneHotEncoder, ColumnTransformer

on_enc = OneHotEncoder(categories='auto')
# OneHotEncoder를 이용하여 생각한다
col_trans = ColumnTransformer([('on_enc'), on_enc, [0]])], remainder='passthrough')

print(col_trans.fit_transform(X))

'''
출력 결과
array([
    [0.0, 1.0, 0.0, 1, 10.1],
    [0.0, 0.0, 1.0, 2, 13.5],
    [1.0, 0.0, 0.0, 3, 15.3],
], dtype=object)
'''
```

---

> **판다스의 get_dummies 메서드를 사용한 원-핫 인코딩**

```python
pd.get_dummies(df[['price', 'color', 'size']])

'''
출력 결과
    price    size     color_blue   color_green   color_red 
0   10.1       1           0            1             0    
1   13.5       2           0            0             1
2   15.3       3           1            0             0
'''
```

**다중 공선성**

~~~
원-핫 인코딩된 데이터셋을 사용할 때 다중 공선성 문제를 유념해야 한다.

특성 간의 상관관계가 높으면 역행렬을 계산하기 어려워 수치적으로 불안정 해진다.

변수 간의 상관관계를 감소하려면 원-핫 인코딩된 배열에서 특성 열 하나를 삭제한다.

예를 들어 color_blue열을 삭제해도 샘플이 color_green = 0이고 color_red = 0 일 때 blue임을 알 수 있다.
~~~

`get_dummies`를 사용할 때 `drop_first` 매개변수를 `True`로 지정하여 첫번째 열을 삭제할 수 있다!

```python
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)

'''
출력 결과
    price    size      color_green   color_red 
0   10.1       1            1             0    
1   13.5       2            0             1
2   15.3       3            0             0
'''
```