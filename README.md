# Scikit-learn MNIST classification using SVM

## Installation

`pip install -r requirements.txt`

# Experiment

MNIST Dataset: train: 10000, test: 60000

1. Find best hyper parameter **C** = [0.01, 0.1, 1, 10, 100, 1000], **transformer** = [normal, scalar, both], **n_split(k)** = [5, 10] with stratified k fold cross validation. 

   => **result_hyparams_k_transf**.csv

2. Train a model with best C each transformer with stratified k fold cross validation.  

   => **scores_k_transf_C**.csv

3. Test a model with test set => **test_score_k_transf_C**.txt

(C: default = 0.01)

## Main Code

```python
def train_model(X, Y):
    normal = Normalizer()
    scalar = StandardScaler()
    clf = svm.SVC(kernel='linear')
	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    pipeline = Pipeline([('transf_normal', normal)('transf_scalar', scalar),  ('estimator', clf)])
    pipeline.set_params(estimator__C=0.01)
    
    scores = cross_validate(pipeline, X, Y, cv=skf, n_jobs=-1,  scoring=["accuracy", "f1_macro"])
```



pipeline을 이용하면 전처리 객체와 분류 모형을 합칠 수 있습니다.

매 fit, predict 마다 step 순서대로 transformer들이 fit, transform 한 후 estimator가 fit, predict 합니다.

따라서 cross validation에서도 validate set이 아닌 train set만 transform할 수 있습니다.

transformer로는 normalizer 또는 Standard Scaler 혹은 둘 다를 사용했습니다.

multi-class 이기 때문에 Stratified K Fold 를 사용했습니다.

```
Pipeline(memory=None,
         steps=[('transf_normal', Normalizer(copy=True, norm='l2')),
                ('transf_scale',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('estimator',
                 SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
                     decision_function_shape='ovr', degree=3,
                     gamma='auto_deprecated', kernel='linear', max_iter=-1,
                     probability=False, random_state=None, shrinking=True,
                     tol=0.001, verbose=True))],
         verbose=False)

```



## Reference

* How to choose best K in K-Fold Cross-Validation
> k = 5 or 10 [ref1](https://datascience.stackexchange.com/questions/28158/how-to-calculate-the-fold-number-k-fold-in-cross-validation) [ref2](https://machinelearningmastery.com/k-fold-cross-validation/)
