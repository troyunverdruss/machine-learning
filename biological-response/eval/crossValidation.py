import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from logLoss import llfun


def main():
    dataset = np.genfromtxt('data/train.csv', dtype='f8', delimiter=',')[1:]

    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])

    cfr = RandomForestClassifier(n_estimators=100)

    cv = KFold(n_splits=5)

    results = []
    for traincv, testcv in cv.split(train):
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append(llfun(target[testcv], [x[1] for x in probas]))

    print('Results: ' + str(np.array(results).mean()))


if __name__ == '__main__':
    main()
