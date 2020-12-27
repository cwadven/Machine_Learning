## 누락된 데이터 다루기

### 테이블 형태 데이터에서 누락된 값 식별

#### 1. CSV 파일 읽기 (문자열 -> CSV : StringID)

```python
import pandas as pd
from io import StringID

# 문자열
csv_data = '''A,B,C,D
1.0,2.0,3.0,4,0
5.0,6.0,7.0,8.0
10.0,11.0,12.0,13.0'''

# StringID를 이용하여 문자열을 CSV 파일 처럼 읽게하기
df = pd.read_csv(StringID(csv_data))
```

#### 2. 읽은 CSV 파일에 누락된 값을 찾기

```python
# isnull 메서드는 셀이 수치 값을 담고 있는지 또는 누락되어 있는지 나타내는 메서드
# sum 메서드 같이 사용하면 해당 열에 있는 누락된 값의 개수를 알 수 있다.
df.isnull().sum()
```

#### 3. 누락된 값을 어떤 방식으로 해결할지 고민 (변환기 클래스 사용)

> **1. 데이터셋에서 해당 샘플(행)이나 특성(열)을 완전히 삭제하기**

**단점**<br>
너무 많은 데이터를 제거하면 안정된 분석이 불가능 할 수 있다.<br>
너무 많은 특성 열을 제거하면 분류기가 클래스를 구분하는 데 필요한 중요한 정보를 잃을 위험이 있다.


```python
'''
이용할 df 내용
    A    B    C   D
0  1.0  2.0  3.0 4.0
1  5.0  6.0  NaN 8.0
2 10.0 11.0 12.0 NaN
'''

# axis = 0 : 행
# axis = 1 : 열

# 행에 하나라도 NaN 값이 있을 경우 제거 시킨다
df.dropna(axis=0)

'''
결과
    A    B    C   D
0  1.0  2.0  3.0 4.0
'''

# 열에 하나라도 NaN 값이 있을 경우 제거 시킨다
df.dropna(axis=1)

'''
결과
    A    B
0  1.0  2.0
1  5.0  6.0
2 10.0 11.0
'''

# 다양한 dropna 기능
# 모든 열이 NaN일 때만 행을 삭제
# (여기서는 모든 값이 NaN인 행이 없기 때문에 원 상태 유지)
df.dropna(how='all')
'''
결과
    A    B    C   D
0  1.0  2.0  3.0 4.0
1  5.0  6.0  NaN 8.0
2 10.0 11.0 12.0 NaN
'''

# 값이 네 개보다 작은 행을 삭제
# (특성이 4개일 경우 3개만 잘나오고 1개가 Nan이면 삭제 느낌)
df.dropna(thresh=4)
'''
결과
    A    B    C   D
0  1.0  2.0  3.0 4.0
'''

# 특정 열에 NaN이 있는 행만 삭제한다(여기서는 'C'열)
df.dropna(subset=['C'])
'''
결과
    A    B    C   D
0  1.0  2.0  3.0 4.0
2 10.0 11.0 12.0 NaN
'''
```

> **2. 보간 기법 사용**

**누락된 값 대체**

- (방법1) 평균으로 대체

각 특성 열의 전체 평균으로 누락된 값을 바꾸는 것<br>
(사이킷런의 SimpleImputer 클래스를 사용)

```python
from sklearn.impute import SimpleImputer

'''
사용할 데이터
    A    B    C   D
0  1.0  2.0  3.0 4.0
1  5.0  6.0  NaN 8.0
2 10.0 11.0 12.0 NaN
'''

# (행 기준)
# strategy : median 또는 most_frequent 이용 가능
# median : 중앙값
# most_frequent : 가장 많이 나타난 값
# constant : 채우고 싶은 값 설정 --> constant 이용 시, fill_value="채우고 싶은것 쓰기"
simr = SimpleImputer(missing_values=np.nan, strategy='mean')
simr = imr.fit(df.values)
imputed_data = imr.transform(df.values)

print(imputed_data)

'''
출력값
array([[1., 2., 3., 4,]
        [5., 6., 7.5, 8.]
        [10., 11., 12., 6.]])
'''

# (열 기준)
# SimpleImputer에는 axis 매개변수가 없으므로 열 기준으로 하고 싶은 경우 FunctionTransformer를 사용하여 행과열의 위치를 바꾼 후
# 해당 바뀐 열 -> 행의 행을 기준으로 SimpleImputer를 이용하고 다시 원 상태로 바꾼다

from sklearn.preprocessing import FunctionTransformer

ftr_simr = FunctionTransformer(lambda X: simr.fit_transform(X.T).T, validate=False)
imputed_data = ftr_simr.fit_transform(df.values)
imputed_data

'''
출력값
array([[1., 2., 3., 4,]
        [5., 6., 6.3333, 8.]
        [10., 11., 12., 11.]])
'''

```