## 텐서플로우 이용하기

> **1. 설치**

~~~
pip install tensorflow
~~~

> **2. 텐서플로우 모듈 import**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 아달라인 모델 같은걸 쓸 것인가? 무엇을 쓸 것인가 하는 것
# 경사하강법
from tensorflow.keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

# one-hot 인코딩 하는 함수
from tensorflow.keras.utils import to_categorical
```