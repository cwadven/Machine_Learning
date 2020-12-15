## 기본 용어와 표기법 소개

### 기본 용어

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/data_set.PNG" alt="drawing" width="600"/><br>

**데이터셋**

데이터들의 집합 (표)

**샘플**

데이터셋에서 하나의 행

**특성**

데이터셋에서 열

**클래스 레이블(타겟)**

내가 알고 싶은 값

---

### 표기법

#### 1.
일반적인 관례에 따라서 `샘플은 행렬 X에 있는 행`으로 나타내고, `특성은 열`을 따라 저장합니다.

150개의 샘플과 네 개의 특성을 가진 데이터셋은 150 X 4 크기의 행렬로 쓸수 있다.

$$X\in R^{150*4}$$

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/matrix.PNG" alt="drawing" width="300"/><br>

#### 2.
굵은소문자는 백터를 나타내고,

$$x\in R^{150*4}$$

굵은 대문자는 행렬을 나타낸다

$$X\in R^{150*4}$$

#### 3.
백터나 행렬에 있는 하나의 원소를 나타낼 때는 이탤릭체를 사용

$$x^{(n)}$$ $$X^{(n)}_{(m)}$$

예제)
$$x^{(행)}_{(열)}$$
$$x^{150}_{1}$$
150번째 샘플(행)의 1번째 차원(열)

#### 4. 특성 행렬
특정 하나의 꽃 샘플을 나타내고 4차원 행 벡터로 쓸 수 있다.

$$x^{(i)}\in R^{1*4}$$

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/row.PNG" alt="drawing" width="300"/><br>

#### 5. 특성 차원

150차원의 열 벡터 

$$x_{j}\in R^{150*1}$$

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/col.PNG" alt="drawing" width="150"/><br>

#### 6. 타깃 변수(클래스 레이블)

150차원의 열 벡터로 저장

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/target.PNG" alt="drawing" width="150"/><br>