## 다층신경망과 텐서플로우

### 신경망 연결

~~~
입력층 --> 은닉층 --> 출력층
~~~

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq1.PNG" alt="drawing" width="500"/><br><br>

### 신경망의 종류

- 단측 퍼셉트론 : 퍼셉트론, 아달라인, 로지스틱 회귀분석
- 다층 퍼셉트론
- 딥러닝 : CNN (다층 퍼셉트론 여러개)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq2.PNG" alt="drawing" width="500"/><br><br>

### 역전파 알고리즘

역전파 알고리즘 : 출력층의 에러를 이용하여 역으로 연결 강도 조정

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq3.PNG" alt="drawing" width="500"/><br><br>

여러개를 들어가서 미분을 한다

#### 수학 공식으로는 알기 어렵기 때문에 데이터를 이용해서 보기

다층 신경망은 Output들이

초기 가중치 랜덤으로 갖는다.<br>
(각각의 노드마다 초기 가중치를 할당 한다, 이때 초기 가중치는 뻗는 노드 개수 만큼 한 노드가 가지고 있다...)

학습률 : 0.2

Output이 클래스 별 하나씩 만들어 진다.

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq4.PNG" alt="drawing" width="500"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq5.PNG" alt="drawing" width="500"/><br><br>

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq6.PNG" alt="drawing" width="500"/><br><br>

비용함수를 이용해서 가중치를 최신화 (미분)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq7.PNG" alt="drawing" width="500"/><br><br>

역전파 즉, 미분을 계속 (편미분)

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq8.PNG" alt="drawing" width="500"/><br><br>


> **잠시 미분 설명**

미분<br>
<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq9.jpg" alt="drawing" width="500"/><br><br>

합성함수의 미분 (함수안에 함수)<br>
<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq10.PNG" alt="drawing" width="500"/><br><br>

편미분 : 필요한 함수만 미분<br>
나는 x만 가지고 미분할래!!! 다른 y 같은 식은 빼고

<img src="https://github.com/cwadven/Machine_Learning/blob/master/ML/chapter12/img/seq11.jpg" alt="drawing" width="500"/><br><br>

