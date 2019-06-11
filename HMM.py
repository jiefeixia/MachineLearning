import numpy as np
import sys
from collections import Counter

train_input = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
max_line = int(sys.argv[4])
predicted_file = sys.argv[5]
metric_file = sys.argv[6]


def load(file):
    X = []
    y = []
    x_y = []
    y_y = []
    with open(file) as f:
        for i, line in enumerate(f):
            l = line.strip().split(" ")
            x, tag = zip(*(s.split("_") for s in l))
            X.append(x)
            y.append(tag)
            x_y += l
            y_y += [tag[i] + "_" + tag[i + 1] for i in range(len(tag) - 1)]

            if i == max_line:
                break

    return np.array(X), np.array(y), x_y, y_y


def logsumexp(logs):
    """
    In case of small float overflow
    """
    logs = logs[logs != -np.inf]
    if logs.shape[0] == 1:
        return logs[0]
    else:
        max_idx = np.argmax(logs)
        max_val = logs[max_idx]
        logs = np.delete(logs, max_idx)
        return max_val + np.log(1 + np.sum(np.exp(logs - max_val)))


if __name__ == '__main__':
    X, y, x_y, y_y = load(train_input)
    with open(index_to_word) as f:
        index_to_word = [word.strip() for word in f.readlines()]
    with open(index_to_tag) as f:
        index_to_tag = [tag.strip() for tag in f.readlines()]

    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
    tag_to_index = {tag: idx for idx, tag in enumerate(index_to_tag)}

    total_x = len(index_to_word)
    total_y = len(index_to_tag)
    prior = np.zeros(total_y)
    emit = np.zeros((total_y, total_x))
    trans = np.zeros((total_y, total_y))

    # prior
    prior_count = Counter([label[0] for label in y])
    for tag_idx in range(total_y):
        prior[tag_idx] = prior_count.get(index_to_tag[tag_idx], 0) + 1
    prior /= np.sum(prior)

    # emit
    emit_count = Counter(x_y)
    for tag_idx in range(total_y):
        for word_idx in range(total_x):
            x_y_key = "%s_%s" % (index_to_word[word_idx], index_to_tag[tag_idx])
            emit[tag_idx, word_idx] = emit_count.get(x_y_key, 0) + 1
    emit = (emit.T / np.sum(emit, axis=1)).T

    # trans
    trans_count = Counter(y_y)
    for y1_idx in range(total_y):
        for y2_idx in range(total_y):
            y_y_key = "%s_%s" % (index_to_tag[y1_idx], index_to_tag[y2_idx])
            trans[y1_idx, y2_idx] = trans_count.get(y_y_key, 0) + 1
    trans = (trans.T / np.sum(trans, axis=1)).T


    # forward backward algorithm
    pi = np.log(prior)
    a = np.log(trans)  # (total_y1, total_y2)
    b = np.log(emit)  # (total_y, total_x)

    # forward
    alphas = []
    for x, tags in zip(X, y):
        alpha = np.zeros((total_y, len(x)))
        for tag in index_to_tag:
            alpha[:, 0] = pi + b[:, word_to_index[x[0]]]
        for t in range(1, len(x)):
            alpha[:, t] = b[:, word_to_index[x[t]]] \
                          + np.array([logsumexp(alpha[:, t - 1] + a[:, y2]) for y2 in range(total_y)])
        alphas.append(alpha)

    # backward
    betas = []
    for x, tags in zip(X, y):
        beta = np.zeros((total_y, len(x)))
        beta[:, -1] = 0
        for t in range(len(x) - 2, -1, -1):
            beta[:, t] = np.array([logsumexp(b[:, word_to_index[x[t + 1]]]
                                             + beta[:, t + 1] + a[y1, :]) for y1 in range(total_y)])
        betas.append(beta)

    # Predict
    preds = []
    ll = []
    acc_sum = 0
    cnt = 0

    for alpha, beta, x, tags in zip(alphas, betas, X, y):
        pred = []
        for t, tag in enumerate(tags):
            y_pred = index_to_tag[np.argmax(alpha[:, t] + beta[:, t])]
            pred.append(y_pred)
            cnt += 1
            if y_pred == tag:
                acc_sum += 1
        preds.append(pred)
        ll.append(logsumexp(alpha[:, t]))
    acc = acc_sum / cnt
    ll = np.average(ll)
    print(ll)

    # save result
    with open(predicted_file, "w") as f:
        for x, pred in zip(X, preds):
            f.write(" ".join([x[i] + "_" + pred[i] for i in range(len(x))]))
            f.write("\n")

    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood : " + str(ll) + "\n")
        f.write("Accuracy : " + str(acc) + "\n")

