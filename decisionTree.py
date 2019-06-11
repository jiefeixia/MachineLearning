import numpy as np
import sys
from collections import Counter


class Node:
    def __init__(self, parent_depth):
        self.depth = parent_depth + 1

        self.left = None
        self.left_edge = None

        self.right = None
        self.right_edge = None

        self.feature = None
        self.y = None

        self.label = None


def split(node, X, y, feature_names):
    # nested function
    def prob(instance):
        freq_value = np.array(list(freq(instance).values()))
        return freq_value / sum(freq_value)

    def freq(instance):
        if len(instance.shape) == 1:
            freq_dict = Counter(instance)
        else:  # for one feature and one label
            freq_dict = Counter(tuple(feature_and_y) for feature_and_y in instance)
        return freq_dict

    def entropy(probs):
        en = 0
        for p in probs:
            en -= p * np.log2(p)
        return en

    # first set node as leaf
    node.y = freq(y)
    for label in node.y.keys():
        if node.y[label] >= len(y) / 2:
            node.label = label

    if node.depth < max_depth and X.size > 0 and len(node.y) > 1:
        # feature selection
        max_mutual_info = 0
        max_f_id = -1
        for f_id, feature in enumerate(X.T):
            mutual_info = entropy(list(prob(feature))) + entropy(list(prob(y))) - \
                          entropy(prob(np.vstack((feature, y)).T))
            if mutual_info > max_mutual_info:
                max_mutual_info = mutual_info
                max_f_id = f_id

        if max_f_id >= 0:  # set node as parent
            node.label = None
            node.feature = feature_names[max_f_id]

            [node.left_edge, node.right_edge] = np.unique(X[:, max_f_id])
            left_examples = X[:, max_f_id] == node.left_edge
            right_examples = X[:, max_f_id] == node.right_edge

            other_features = np.arange(X.shape[1]) != max_f_id
            node.left = split(Node(node.depth), X[left_examples, :][:, other_features], y[left_examples],
                              feature_names[other_features])
            node.right = split(Node(node.depth), X[right_examples, :][:, other_features], y[right_examples],
                               feature_names[other_features])

    return node


def predict(X, root):
    y = []
    for x in X:
        node = root
        while not node.label:  # node is not leaf
            if x[feature_names[:-1] == node.feature] == node.left_edge:
                node = node.left
            else:
                node = node.right
        y.append(node.label)

    return np.array(y)


def metrics(pred, y):
    return np.sum([pred != y]) / len(y)


def print_node(node):
    if not node.label:  # node is not leaf
        print("| " * (node.depth + 1) + node.feature + " = " + node.left_edge + ": " + print_label(node.left))
        print_node(node.left)

        print("| " * (node.depth + 1) + node.feature + " = " + node.right_edge + ": " + print_label(node.right))
        print_node(node.right)


def print_label(node):
    s = "["
    for label in np.unique(y_train):
        if label in node.y.keys():
            s += str(node.y[label]) + " " + label + "/"
        else:
            s += "0" + " " + label + "/"
    s = s[:-1] + "]"
    return s


if __name__ == '__main__':
    # train = "education_train.csv"
    # test = "education_test.csv"
    # # train = "small_train.csv"
    # # test = "small_test.csv"
    # max_depth = 3

    train = sys.argv[1]
    test = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    with open(train) as f:
        data = []
        for line in f:
            data.append(line.split(","))
        data = np.array(data)
        data[:, -1] = [label.rstrip() for label in data[:, -1]]
        feature_names = data[0, :]
        X_train = data[1:, :-1]
        y_train = data[1:, -1]
    with open(test) as f:
        data = []
        for line in f:
            data.append(line.split(","))
        data = np.array(data)
        data[:, -1] = [label.rstrip() for label in data[:, -1]]
        X_test = data[1:, :-1]
        y_test = data[1:, -1]

    root = split(Node(-1), X_train, y_train, feature_names[:-1])

    # print("=============depth: %s==============" % max_depth)
    print(print_label(root))
    print_node(root)

    train_pred = predict(X_train, root)
    test_pred = predict(X_test, root)
    train_error = metrics(train_pred, y_train)
    test_error = metrics(test_pred, y_test)

    with open(train_out, "w") as f:
        for label in train_pred:
            f.write(label+"\n")

    with open(test_out, "w") as f:
        for label in test_pred:
            f.write(label+"\n")

    with open(metrics_out, "w") as f:
        # print("error(train): %f" % train_error)
        # print("error(test): %f" % test_error)

        # print("%f %f" % (train_error, test_error))

        f.write("error(train): %f\n" % train_error)
        f.write("error(test): %f" % test_error)
