import pandas as pd
from numpy import savetxt

import shared


def make_int(x):
    if x >= 0.50:
        return 1
    else:
        return 0


def main():
    # http://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb

    df = shared.load_data()
    train = shared.get_train(df)
    target = shared.get_target(df)

    orig_test_df = pd.read_csv('data/test.csv')
    test_df = shared.preprocess_data(orig_test_df)

    rf = shared.get_model()
    rf.fit(train, target)

    predicted_probs = [[orig_test_df['PassengerId'][index], make_int(x[1])] for index, x in
                       enumerate(rf.predict_proba(test_df))]

    savetxt('out/submission.csv', predicted_probs, delimiter=',', fmt='%d,%d',
            header='PassengerId,Survived', comments='')

    print('See out/submission.csv for results')


if __name__ == "__main__":
    main()
