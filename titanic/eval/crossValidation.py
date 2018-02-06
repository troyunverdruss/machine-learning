import sys

import numpy as np
from sklearn.model_selection import KFold

from logLoss import llfun

sys.path.insert(0, 'src/')
sys.path.insert(0, '../src/')
import shared


def main():
    df = shared.load_data()
    train = shared.get_train(df)
    target = shared.get_target(df)

    rf = shared.get_model()

    cv = KFold(n_splits=5)

    results = []
    for traincv, testcv in cv.split(train):
        probas = rf.fit(train.iloc[traincv], target.iloc[traincv]).predict_proba(train.iloc[testcv])
        results.append(llfun(target.iloc[testcv], [x[1] for x in probas]))

    print('Results: ' + str(np.array(results).mean()))


if __name__ == '__main__':
    main()
