import numpy as np
import sys
from scipy.sparse import csc_matrix

formatted_train_input = sys.argv[1]
formatted_validation_input = sys.argv[2]
formatted_test_input = sys.argv[3]
dict_input = sys.argv[4]
train_out = sys.argv[5]
test_out = sys.argv[6]
metrics_out = sys.argv[7]
num_epoch = int(sys.argv[8])

with open(dict_input) as f:
    FEATURE_NUM = len(f.readlines())

LEARNING_RATE = 0.1


class LogisticRegression:
    def __init__(self, feature_num):
        self.w = np.zeros(feature_num + 1)

    def pred(self, x):
        return 1 / (1 + np.exp(- x.dot(self.w)))

    def derivative(self, x, y):
        """
        calculate dJ/d(theta) for ith example
        :param x: vetor of length (feature num + 1)
        :param y: scalar
        :return: vetor of length (feature num + 1)
        """
        return x * (y - self.pred(x))[0]

    def step(self, x, y, lr):
        self.w += lr * self.derivative(x, y).toarray()[0]


def nllloss(pred, y):
    return np.average(- np.log(y * pred + (1 - y) * (1 - pred)))


def read(formatted_input):
    labels = []
    col = []
    row = []
    with open(formatted_input) as f:
        for row_idx, line in enumerate(f):
            contents = line.strip().split("\t")
            labels.append(int(contents[0]))
            # x0 is for bias parameter
            col += [0]
            row += [row_idx]

            row += [row_idx for _ in contents[1:]]
            col += [int(content[:-2]) + 1 for content in contents[1:]]

    return csc_matrix(([1 for _ in range(len(col))], (row, col)), shape=[row[-1] + 1, FEATURE_NUM + 1]), \
           np.array(labels)


if __name__ == '__main__':
    x_train, y_train = read(formatted_train_input)
    x_test, y_test = read(formatted_test_input)
    x_val, y_val = read(formatted_validation_input)

    lg = LogisticRegression(FEATURE_NUM)

    nlls_train = []
    nlls_val = []
    for e in range(num_epoch):
        print("epoch %d" % e, end=",")
        for i in range(x_train.shape[0]):
            lg.step(x_train[i], y_train[i], LEARNING_RATE)
        nll_train = nllloss(lg.pred(x_train), y_train)
        nll_val = nllloss(lg.pred(x_val), y_val)

        print("%f,%f" % (nll_train, nll_val))
        nlls_train.append(nll_train)
        nlls_val.append(nll_val)

    pred_train = lg.pred(x_train)
    pred_test = lg.pred(x_test)

    pred_train[pred_train > 0.5] = 1
    pred_train[pred_train <= 0.5] = 0
    pred_test[pred_test > 0.5] = 1
    pred_test[pred_test <= 0.5] = 0

    error_train = 1 - np.sum(pred_train == y_train) / len(pred_train)
    error_test = 1 - np.sum(pred_test == y_test) / len(pred_test)

    with open(train_out, "w", encoding="utf8") as f:
        for label in pred_train:
            f.writelines("%d\n" % label)

    with open(test_out, "w", encoding="utf8") as f:
        for label in pred_test:
            f.writelines("%d\n" % label)

    with open(metrics_out, "w") as f:
        f.write("error(train): %f\n" % error_train +
                "error(test): %f\n" % error_test)
