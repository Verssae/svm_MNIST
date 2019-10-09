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

    def get_img(idx): return np.concatenate(([lbl[idx]],list(img[idx])), axis=0)

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
    # scalar = StandardScaler()
    clf = svm.SVC(kernel='linear')
    pipeline = Pipeline([('trnsf_normal',normal),  ('estimator', clf)])
    pipeline.set_params(estimator__C=1, estimator__gamma=1e-3)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(pipeline, X, Y, cv=skf, n_jobs=-1, verbose=True)
    print(scores.mean())
    # print("[Saving model]")
    # joblib.dump(svmc, 'saved_model.pkl')


def test():
    X, Y = get_data("testing")
    print("[Loading model]")
    loaded_model = joblib.load('saved_model.pkl')
    print("[Predicting]")
    predictions = loaded_model.predict(X)
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    result = metrics.classification_report(
        y_true=Y, y_pred=predictions, target_names=target_names)
    print(result)


def find_best_hyparms(X, Y):
    normal = Normalizer()
    clf = svm.SVC(kernel='linear')
    pipeline = Pipeline([('trnsf_normal',normal),  ('estimator', clf)])
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    parameters = {'estimator__C': [1, 10, 100],
                  'estimator__gamma': [1e-2, 1e-3, 1e-4]}

    print("[Finding hyparmas]")

    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, return_train_score=True, cv=skf, verbose=True)
    clf.fit(X, Y)

    cv_results = pd.DataFrame(clf.cv_results_)
    return cv_results


def plot_result(result):
    # converting C to numeric type for plotting on x-axis
    result['param_C'] = result['param_C'].astype('int')

    # # plotting
    plt.figure(figsize=(16, 6))

    # subplot 1/3
    plt.subplot(131)
    gamma_01 = result[result['param_gamma'] == 0.01]

    plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
    plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title("Gamma=0.01")
    plt.ylim([0.60, 1])
    plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
    plt.xscale('log')

    # subplot 2/3
    plt.subplot(132)
    gamma_001 = result[result['param_gamma'] == 0.001]

    plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
    plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title("Gamma=0.001")
    plt.ylim([0.60, 1])
    plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
    plt.xscale('log')


    # subplot 3/3
    plt.subplot(133)
    gamma_0001 = result[result['param_gamma'] == 0.0001]

    plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
    plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title("Gamma=0.0001")
    plt.ylim([0.60, 1])
    plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
    plt.xscale('log')

    plt.show()

if __name__ == "__main__":
    # get_data()
    choice = int(input("1: train 2: load model and test 3: finding hyparams: "))
    if choice == 1:
        main()
    if choice == 2:
        test()
    if choice == 3:
        X, Y = get_data()
        result = find_best_hyparms(X,Y)
        f = open('result.txt', 'w')
        f.write(result.to_string())
        f.write("\nDESCR:\n")
        f.write(result.describe().to_string())
        print(result)
        print(result.describe())
        # plot_result(result)
