## 머신 러닝을 위한 파이썬

> **NumPy(넘파이), SciPy(사이파이)**
포트란과 C언어로 만든 저수준 모듈위에 구축된 라이브러리

### 환경 구축

- **Python Version 3.7.2**


#### VSCODE 사용시

1. 가상환경 생성
~~~
python -m venv myvenv(생성할가상환경이름)
~~~

2. 가상환경 실행
~~~
source myvenv/Script/activate [윈도우 버전]
source myvenv/bin/activate [리눅스 버전]
~~~

3. 모듈 설치
~~~
pip install 모듈 이름
pip install 모듈 이름
pip install 모듈 이름 ....

혹은

pip install -r requirements.txt
(txt 파일에 모듈과 버전이 들어있는 파일)
~~~

---

#### Anaconda 사용시

1. 가상환경 생성 (python 3 설치)
~~~
conda create -n myvenv(가상환경이름) python=3.7.2
~~~

2. 가상환경 시작
~~~
source actiavte myvenv(가상환경이름) [리눅스 버전]
actiavte myvenv(가상환경이름) [윈도우 버전]
~~~

3. 모듈 설치
~~~
pip install 모듈 이름
pip install 모듈 이름
pip install 모듈 이름 ....

혹은

pip install -r requirements.txt
(txt 파일에 모듈과 버전이 들어있는 파일)
~~~

---

### 과학 컴퓨팅, 데이터 과학, 머신 러닝을 위한 패키지

- NumPy 1.16.1 (다차원 배열)
- SciPy 1.2.1 (과학 계산용 함수)
- Scikit-learn 0.20.2 (인공지능 알고리즘)
- Matplotib 3.0.2 (그래프)
- Pandas 0.24.1 (표)
- TensorFlow 2.0.0 (인공지능 알고리즘)