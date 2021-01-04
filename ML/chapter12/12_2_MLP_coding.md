## sklearn 다층신경망 코딩

```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

# 얼마나 예측 잘 했는지 확인 하기 위해서 사용
from sklearn.metrics import accuracy_score

## 6.모델검정
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # 정확도, 민감도 등
from sklearn.metrics import roc_curve, auc # ROC 곡선 그리기

## 7.최적화
from sklearn.model_selection import cross_validate # 교차타당도
from sklearn.pipeline import make_pipeline # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV # 

# csv 파일 가져오기
dataset = pd.read_csv("ML/chapter3/accidentsnn.csv")

# X랑 y 분리 시키기
X = dataset.drop(["MAX_SEV_IR"], axis=1)
y = dataset["MAX_SEV_IR"]


X["ALCHL_I"] = X["ALCHL_I"].replace([1,2], ['Yes', 'No'])
X['PROFIL_I_R'] = X['PROFIL_I_R'].replace([0,1], ['etc', 'level1'])
X["SUR_COND"] = X["SUR_COND"].replace([1,2,3,9], ['dry', 'wet', 'snow', 'non'])


# 수치화된 데이터를 가변수화
# 가변수화를 하는 이유는 만약 1:월, 2:화, 3:수 일 경우 1 + 2 = 3 이기에 수요일과 관계가 생길 수 있기 때문
# get_dummies(dataframe, dataframe에 가변수화 할 열, drop_first=True)
# drop_first : 가변수화할 때 이상한 값이 나오게 됩니다! 그거 막기 위함
X = pd.get_dummies(X[["ALCHL_I", "PROFIL_I_R", "VEH_INVL", "SUR_COND"]],
columns=["ALCHL_I", "SUR_COND", "PROFIL_I_R"], drop_first=True)

# 클래스 또한 더미로 만든다
y = pd.get_dummies(y)

# 전처리를 했으니 학습용 데이터와 검증용 데이터 분류
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 분리 시켜줄 것 생각해 보기
sc = StandardScaler()

# Standard 적용한 값 슬라이싱을 통해 재할당 하기!
# X_train.iloc[[행], [열]] : 이름으로 접근 / X_train.iloc[[행], [열]] : index로 접근
# X_test.iloc[[행], [열]] : 이름으로 접근 / X_test.iloc[[행], [열]] : index로 접근
X_train.iloc[:, [0]] = sc.fit_transform(X_train.iloc[:, [0]])
X_test.iloc[:, [0]] = sc.fit_transform(X_test.iloc[:, [0]])

# MLP 모델 구축
# hidden_layer_size (첫번째레이어노드개수, 두번째레이어노드개수, 세번째레이어노드개수 ....)
mlp = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(10,10),
    random_state=1
)
mlp.fit(X_train, y_train)

# 모델 검증
# 클래스를 분리했던 것을 다시 원상태로 만들기 위해서
y_pred = pd.DataFrame(mlp.predict(X_test))
y_pred = y_pred.idxmax(axis=1)


y_test = y_test.idxmax(axis=1)

# 정오 분류표 만들기
confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        index=['True[0]', 'True[1]', 'True[2]'],
                        columns=['Predict[0]', 'Predict[1]', 'Predict[2]'])

print('정확도: %.2f' % accuracy_score(y_test, y_pred))

# 최적화를 하기 위한 하이퍼파라미터 구하기
# 파이프라인 모델 구축
pipe_mlp = make_pipeline(MLPClassifier(random_state=1))
pipe_mlp.get_params().keys()

# 학습 곡선으로 편향과 분산 문제 분석하기
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_mlp,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()


# 검증 곡선으로 과대적합과 과소적합 조사하기
param_range = [1e-06, 1e-05, 0.0001, 0.001]  # 수정

train_scores, test_scores = validation_curve(
                estimator=pipe_mlp, # 수정
                X=X_train, 
                y=y_train, 
                param_name='mlpclassifier__alpha', ## 수정
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of C') # 수정
plt.legend(loc='lower right')
plt.xlabel('Parameter C') # 수정
plt.ylabel('Accuracy')
plt.ylim([0.8, 0.9])  # 수정
plt.tight_layout()
plt.show()


# 하이퍼 파라미터 튜닝
param_range1 = [(5,5), (5,10), (10,5), (10, 10)]  # 수정
param_range2 = [1e-06, 1e-05, 0.0001, 0.001]  # 수정

param_grid = [{'mlpclassifier__hidden_layer_sizes': param_range1, # 수정
               'mlpclassifier__alpha': param_range2}] # 수정

gs = GridSearchCV(estimator=pipe_mlp, # 수정
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)
```