from __future__ import division

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier


def train():
    """
    在我们得到的特征上训练分类器，target为1(感兴趣)，或者是0(不感兴趣)
    """
    trainDf = pd.read_csv("data_train.csv")
    trainDf.fillna(0, inplace=True)
    X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco",
                                        "user_pop", "frnd_infl", "evt_pop"]))
    y = np.array(trainDf.interested)
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(X, y)
    return clf


def validate():
    """
    10折的交叉验证，并输出交叉验证的平均准确率
    """
    trainDf = pd.read_csv("data_train.csv")
    trainDf.fillna(0, inplace=True)
    X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco",
                                        "user_pop", "frnd_infl", "evt_pop"]))
    y = np.array(trainDf.interested)
    kfold = KFold(n_splits=10)
    avgAccuracy = 0
    run = 0
    for train, test in kfold.split(X):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf = SGDClassifier(loss="log", penalty="l2")
        clf.fit(Xtrain, ytrain)
        accuracy = 0
        ntest = len(ytest)
        for i in range(0, ntest):
            yt = clf.predict(Xtest[i, :])
            if yt == ytest[i]:
                accuracy += 1
        accuracy = accuracy / ntest
        print("accuracy (run %d): %f" % (run, accuracy))
        avgAccuracy += accuracy
        run += 1
    print("Average accuracy {}", (avgAccuracy / run))


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


trainDf = pd.read_csv("data_train.csv")
trainDf.fillna(0, inplace=True)
X = np.matrix(pd.DataFrame(trainDf, index=None,
                           columns=["invited", "user_reco", "evt_p_reco", "evt_c_reco", "user_pop", "frnd_infl",
                                    "evt_pop"]))
y = np.array(trainDf.interested)
plot_learning_curve(clf, u"学习曲线", X, y, cv=10)
