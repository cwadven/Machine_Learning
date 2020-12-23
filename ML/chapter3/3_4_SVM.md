## 서포트 벡터 머신을 사용항 최대 마진 분류

> **서포트 벡터 머신**

<ul>
<li>강력하고 널리 사용되는 학습 알고리즘.</li>
<li>퍼셉트론의 확장</li>
<li>마진을 최대화하는 것</li>
</ul>

마진은 클래스를 구분하는 초평면(결정 경계)과 이 초평면에 가장 가까운 훈련 샘플 사이의 거리로 정의한다.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter3/img/SVM.PNG" alt="drawing" width="600"/><br><br>

> **최대 마진**

큰 마진의 결정 경계를 원하는 이유는 일반화 오차가 낮아지는 경향이 있기 때문이다.

반면에 작은 마진의 모델은 과대적합되기 쉽다.

~~~
과대적합 : 되면 좋지 못한것...
~~~

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

