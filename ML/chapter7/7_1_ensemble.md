## 앙상블 학습

**앙상블 이란?**

여러 분류기(모델 알고리즘)를 하나의 메타(통합) 분류기로 연결하여 개별 분류기보다 더 좋은 일반화 성능을 달성

**방법**

여러 분류 알고리즘 사용 : 다수결 투표<br>
(다수결 투표 : 여러개 동시에 써서 가장 좋은 모델들을 결합해서 쓰는 방법)

하나의 분류 알고리즘 이용 : 배길, 부스팅<br>
(반복 측정하는 방법)

**종류**

- 투표(voting) : 동일한 훈련세트로 모델 구축

- 배깅(bagging) : 하나의 모델 훈련세트를 여러개 쪼게서<br>
(랜덤 포레스트 : 의사결정 나무, 배기으이 목적으로 만들어진 것)

- 부스팅(boosting) : 샘플 뽑을 때 잘못 분류된 data 50%를 재학습, 또는 가중치 이용

---

> **투표 방법**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter7/img/seq1.PNG" alt="drawing" width="500"/><br><br>

~~~
동일한 훈련세트 하나를 여러개의 모델들을 돌려서 훈련시킨 후, 각각 모델이 예측한 값이 있는데, 그중에서 가장 많이 나온 값을 예측값으로 이용
~~~


> **배깅 방법**

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter7/img/seq2.PNG" alt="drawing" width="600"/><br><br>

~~~
하나의 모델에서 여러가지 다양한 옵션을 적용하여
부트스트랩 실시 : 데이터로 부터 복원추출(중복허용)을 이용
~~~

**배깅의 방법에서 나오는 예 -> 랜덤 포레스트 (배깅의 일종)**

결정 트리를 하나를 여러개 이용해서 트리를 Forest 라고 말을 바꾸는 느낌이다.

여러개의 트리를 만드는데 요인을 바꾸면서 훈련세트를 넣는 것이다.


<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter7/img/seq3.PNG" alt="drawing" width="600"/><br><br>


> **부스팅 방법**

배깅과 비슷한 방법

샘플을 뽑을 때 잘못 분류된 data 50%를 재학습

훈련세트는 반을 다시 쓰고, 새로운 것 반을 쓰고

AdaBoost : 전체 훈련 샘플 사용하고, 잘못 분류된 data에 가중치 적용

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter7/img/seq4.PNG" alt="drawing" width="600"/><br><br>