from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier


def main():
    dataset = genfromtxt('data/train.csv', dtype='f8', delimiter=',')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt('data/test.csv', dtype='f8', delimiter=',')[1:]

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    savetxt('out/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f',
            header='MoleculeId,PredictedProbability', comments='')

    print('See out/submission.csv for results')


if __name__ == "__main__":
    main()
