
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_score_60k():
    c = ['0.01', '0.1', '1', '10', '100', '1000']
    acc = []
    f1 = []
    for val in c:
        df = pd.read_csv(f'60000\\scores_dscr_5_normal_60000_c={val}.csv')
        acc.append(df["test_accuracy"][1])
        f1.append(df["test_f1_macro"][1])
    print(acc)
    print(f1)
    plt.plot(c, acc, 'ob-', label='acc')
    plt.plot(c, f1,'or-', label='f1')
    plt.xlabel('C')
    plt.ylabel('acc , f1')
    plt.title('k=5, dataset=60000, preprocessor=Normalizer')
    plt.legend(loc='upper right')
    plt.show()

def plot_test_60k():
    c = ['0.01', '0.1', '1', '10', '100', '1000']
    f1 = [0.91, 0.93, 0.95, 0.95, 0.94, 0.93]
    
    # print(acc)
    # print(f1)
    plt.plot(c, f1, 'ob-', label='f1', )
    # plt.plot(c, f1, color='r', label='f1')
    plt.xlabel('C')
    plt.ylabel('f1')
    plt.title('Test f1-score: dataset=60000, preprocessor=Normalizer')
    # plt.legend(loc='upper right')
    plt.show()

def plot_hyparms_10k():
    names_5 = [ '5_normal', '5_scalar','5_both' ]
    names_10 = ['10_normal','10_scalar', '10_both']
    c = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
    styles = ['ob-','or-','og-']
    plt.title('find best C: dataset=10000')
    plt.subplot(1,2,1)
    plt.title('k=5')
    cnt = 0
    for val in names_5:
        df = pd.read_csv(f'result\\result_hyparams_{val}.csv')
        mean_test_score = df["mean_test_score"]
    
        plt.plot(c, mean_test_score, styles[cnt], label=val)
        plt.xlabel('C')
        plt.ylabel('test-score')
        cnt += 1
    plt.legend(loc='lower right')
    
    plt.subplot(1,2,2)
    plt.title('k=10')
    cnt = 0
    for val in names_10:
        df = pd.read_csv(f'result\\result_hyparams_{val}.csv')
        mean_test_score = df["mean_test_score"]
    
        plt.plot(c, mean_test_score,  styles[cnt], label=val)
        plt.xlabel('C')
        plt.ylabel('test-score')
        cnt += 1
    plt.legend(loc='lower right')
    
    plt.show()

def plot_score_10k():
    names = ['5_both_c=0.01','5_normal_c=10','5_scalar_c=0.01','10_normal_c=1']
    styles = ['ob-','or-','og-', 'om-']
    acc = []
    f1 = []
    for val in names:
        df = pd.read_csv(f'result\\scores_dscr_{val}.csv')
        acc.append(df["test_accuracy"][1])
        f1.append(df["test_f1_macro"][1])
    print(acc)
    print(f1)
    plt.plot(names, acc,styles[0], label='acc')
    plt.plot(names, f1,styles[1], label='f1')
 
    plt.title('dataset=10000, best estimator')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    plot_hyparms_10k()