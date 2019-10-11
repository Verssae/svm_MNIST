import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import struct
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, Normalizer
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline


import joblib

name = "10_scalar"

def read(dataset="training", path="MNIST"):
    if dataset is "testing":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "training":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise Exception("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    def get_img(idx): return np.concatenate(
        ([lbl[idx]], list(img[idx])), axis=0)

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def main():
    X, Y = get_data()
    train_model(X, Y)


def get_data(dataset="training"):
    print("[Reading dataset]")
    tr = list(read(dataset))
    columns = ["label"] + [f'#{x}' for x in range(784)]
    df = pd.DataFrame(tr, columns=columns)
    print(df.describe())

    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # X = X / 255.0
    return X, Y


def train_model(X, Y):
    print("[Training Model]")
    normal = Normalizer()
    scalar = StandardScaler()
    clf = svm.SVC(kernel='linear')
    pipeline = Pipeline(
        [ ('transf', normal),('transf2',scalar),('estimator', clf)])
    pipeline.set_params(estimator__C=1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_validate(pipeline, X, Y, cv=skf,
                            n_jobs=-1,  scoring=["accuracy", "f1_macro"])

    scores = pd.DataFrame(scores)
    print(scores)
    scores.to_csv(f'scores_{name}.csv')
    scores.describe().to_csv(f'scores_dscr_{name}.csv')

    fitted_pipeline = pipeline.fit(X, Y)
    print("[Saving model]")
    joblib.dump(fitted_pipeline, f'saved_model_{name}.pkl')




def test():
    X, Y = get_data("testing")
    print("[Loading model]")
    loaded_model = joblib.load(f'saved_model_{name}.pkl')
    print("[Predicting]")
    predictions = loaded_model.predict(X)
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    result = metrics.classification_report(
        y_true=Y, y_pred=predictions, target_names=target_names)
    print(result)
    f = open(f'test_score_{name}.txt', 'w')
    f.write(result)
    f.close()



def find_best_hyparms(X, Y):
    normal = Normalizer()
    scalar = StandardScaler()
    clf = svm.SVC(kernel='linear')
    pipeline = Pipeline(
        [  ('transf2',scalar), ('estimator', clf)])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    parameters = {'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    print("[Finding hyparmas]")

    clf = GridSearchCV(pipeline, param_grid=parameters,
                       n_jobs=-1, cv=skf, verbose=True)
    clf.fit(X, Y)

    cv_results = pd.DataFrame(clf.cv_results_)
    print(clf.best_params_)
    print(clf.best_estimator_)
    print(clf.best_score_)
    return cv_results


if __name__ == "__main__":
    # get_data()
    print(f"{name}")
    choice = int(
        input("1: train 2: load model and test 3: finding hyparams: "))
    if choice == 1:
        main()
    if choice == 2:
        test()
    if choice == 3:
        X, Y = get_data()
        result = find_best_hyparms(X, Y)
        result.to_csv(f'result_hyparams_{name}.csv')
        # result.to_csv('result_hyparams_descr.csv')
        # f = open('result.txt', 'w')
        # f.write(result.to_string())
        # f.write("\nDESCR:\n")
        # f.write(result.describe().to_string())
        # print(result)
        # print(result.describe())
        # plot_result(result)
