## 머신러닝의 세가지 종류

> **지도 학습**

**특징**

~~~
레이블된 데이터
직접 피드백
출력 및 미래 예측
~~~

**지도 학습의 주요 목적**

레이블된 훈련 데이터에서 모델을 학습하여 본 적 없는 미래 데이터에 대해 예츠을 만드는 것

`레이블(label) : 샘플에 할당된 클래스 --> (분류된 결과)`

**지도학습 순서도**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/supervised.PNG" alt="drawing" width="600"/><br>

**◼ 분류 : 클래스 레이블 예측**

- 지도 학습의 하위 카테고리이다.
- 과거의 관측을 기반으로 새로운 샘플의 범주형 클래스 레이블을 예측하는 것이 목표

~~~
이진 분류 : 2개 중 1개 분류
예) 스팸 메일 인지 아닌지

다중 분류 : N개 중 1개 분류
예) 손글씨 인식 각각의 글자 인식
~~~

**이진 분류 작업 개념**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/bit2.PNG" alt="drawing" width="600"/><br>

15개의 샘플은 음성 클래스, 또 다른 15개의 샘플은 양성 클래스로 레이블 되어있다.

지도 학습 알고리즘을 사용하여 두 클래스를 구분할 수 있는 규칙을 학습한다.

이 규칙은 점선으로 나타난 `결정 경계` 이다.

**◼ 회귀 : 연속적인 출력 값 예측**

`예측 변수`(설명 변수, 입력)와 연속적인 `반응 변수`(출력, 타깃)가 주어졌을 때 출력 값을 예측 하는 두 변수 사이의 관계

**회귀의 선형 회귀의 개념**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/linear_regression.PNG" alt="drawing" width="600"/><br>

입력 x와 타깃 y가 주어지면 샘플과 직선 사이 거리가 최고가 되는 직선을 그을 수 있다.

일반적으로 평균 제곱 거리를 사용

이렇게 데이터에서 학습한 직선의 **기울기와 절편**을 사용하여 새로운 데이터의 출력 값을 예측

---

> **비지도 학습**

**특징**

~~~
레이블 및 타겟 없음
피드백 없음
데이터에서 숨겨진 구조 찾기
~~~

**◼ 군집 : 서브그룹 찾기**

그룹 정보를 의미 있는 서브그룹 또는 클러스터로 조직하는 탐색적 데이터 분석 기법.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/cluster.PNG" alt="drawing" width="600"/><br>

하나의 `大 그룹`을 유사한 무엇가를 통해 `小 그룹`으로 나누는 방식.

**◼ 차원 축소 : 데이터 압축**

비지도 차원 축소는 잡음(noise) 데이터를 제거하기 위해 특성 전처리 단계에서 종종 적용하는 방법.

차원 축소는 관련 있는 정보를 대부분 유지하면서 더 작은 차원의 부분 공간으로 데이터를 압축.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/2d_3d.PNG" alt="drawing" width="600"/><br>

---

> **강화 학습**

**특징**

~~~
결정 과정
보상 시스템 구축
연속된 행동에서 학습
~~~

**강화 학습의 주요 목적**

환경과 상호 작용하여 시스템(에이전트) 성능을 향상하는 것이 목적

- 강화 학습을 지도 학습과 관련된 분야로 생각 가능


강화 학습의 피드백은 정답 레이블이나 값이 아니다.<br>
보상 함수로 얼마나 행동이 좋은지를 측정한 값이다.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter1/img/strong.PNG" alt="drawing" width="600"/><br>

각 상태는 양의 보상이나 음의 보상과 연관된다.

체스를 예를 들어서 상대 체스 기물을 잡거나 퀸을 위협하는 것을 근정적인 이벤트로 양의 보상으로 취할 수 있고, 반대로 자신의 체스 기물이 잡히거나 퀸이 위협을 당하면 이벤트를 음의 보상으로 취할 수 있다.