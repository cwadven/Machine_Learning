## 최적화

우리가 기존에 하던 방식들

홀드아웃 방법 : 7대 3으로 쪼개는 경우

### 교차검증(Cross Validation : CV)

과소적합이 아니고 과대적합이 아닌지를 확인 하기 위한 과정

최적화할 모델을 찾기 위해서 하이퍼파라미터를 통해 여러 경우의 수를 알기 위한 방법.

이럴 때 쓰는 테스트 데이터는 훈련 데이터에 있는 것중 9:1 비율로 1로 테스트 데이터로 이용을 한다.<br>
(이것을 할 때도 어느정도 정확한지 알아야하니 테스트가 필요한데 따로 테스트를 안 나누는데 k-겹 교차검증 은 9:1로 나눈다)

훈련 데이터에서 어느정도 빼내서 테스트 데이터로 만든다.

최적화 할 때, 튜닝을 할 때
 
> **과정**

1. 기본적인 모델 세팅 (파이프라인)
2. learning_curve를 이용하여 과소 적합인지 확인
3. validation_curve를 이용하여 최적의 하이퍼파라미터(종류의 최적 값)를 찾기 위해 여러 경우의 수를 넣는다
4. 최적의 하이퍼파라미터를 찾아 그것을 이용하여 모델을 구축한다.

### 파이프라인

하이퍼파라미터를 세팅

기본모형은 아무 옵션이 없는 모델로 시작 (디폴트로)

```python
# 옵션을 넣지 않는다
DecisionTreeClassifier()
```

파이프 라인으로 만들기 (연결)

```python
# 옵션을 넣지 않는다
pipe_tree = make_pipeline(DecisionTreeClassifier())
```

~~~
- make_pipeline(기본모델틀)
~~~

make_pipline 하면 모델을 여러개 쓸 수 있기 때문에 기본 모델에서 반환되는 값들이 다르다.

make_pipline을 통해서 반환하면 해당 DesisionTreeClassifier에 있는 것들이 decisiontreeclassifier__class_weight.... 등 처럼 변한다!

> **파이프라인 모델 구축 순서**

1. 학습 곡선으로 편향과 분산 문제 분석하기 (데이터들이 적당하게 학습이 되었는지 확인하는 그래프 그리기)<br>
학습 몇번을 해야 최소적합이 아니게 되는지 확인하는 그래프

```python
train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_tree, # 수정
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.03])  # 수정
plt.tight_layout()
plt.show()
```

2. 과대적합 부분을 확인하고, 하이퍼파라미터를 몇으로 설정할지 확인

```python
# 어떤 특정 기준을 가지는 것을 경우의 수 별로 나눈다.
param_range = [1,2,3,4,5,6,7,8,9,10]  # 수정

train_scores, test_scores = validation_curve(
                estimator=pipe_tree, # 수정
                X=X_train, 
                y=y_train,
                # 하이퍼 파라미더 기준을 max_depth로 하겠다
                param_name='decisiontreeclassifier__max_depth', ## 수정
                # 경우의 수를 리스트로 대입
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
plt.legend(loc='lower right')
plt.xlabel('Parameter max_depth') # 수정
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.09])  # 수정
plt.tight_layout()
plt.show()
```

3. 위의 것은 하나의 하이퍼파라미터 밖에 쓰지를 못한다 그래서 다른 CV용을 찾는다.

```python
param_range1 = [1,2,3,4,5,6,7,8,9,10] # 수정
param_range2 = [10,20,30,40,50] # 수정

param_grid = [{'decisiontreeclassifier__max_depth': param_range1, # 수정
               'decisiontreeclassifier__min_samples_leaf': param_range2}] # 수정

gs = GridSearchCV(estimator=pipe_tree, # 수정
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)
```