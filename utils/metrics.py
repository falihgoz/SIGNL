import numpy as np
from sklearn.metrics import confusion_matrix


def compute_eer(scores, labels):
    if isinstance(scores, list) is False:
        scores = list(scores)
    if isinstance(labels, list) is False:
        labels = list(labels)

    target_scores = []
    nontarget_scores = []

    for item in zip(scores, labels):
        if item[1] == 1:
            target_scores.append(item[0])
        else:
            nontarget_scores.append(item[0])

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    if target_size == 0 or nontarget_size == 0:
        print("Target size: ", target_size)
        print("non-target size: ", nontarget_size)
        raise ValueError("Target or non-target scores are empty.")

    target_position = 0
    for i in range(target_size - 1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break

    if target_position >= len(target_scores):
        raise IndexError(
            f"Target position {target_position} is out of bounds for target_scores of size {len(target_scores)}."
        )

    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th


def compute_metrics(labels, scores):
    # try:
    eer, th = compute_eer(scores, labels)
    preds = np.where(scores >= th, 1, 0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp + 1e-4)
    recall = tp / (tp + fn + 1e-4)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-4)

    print("True Negatives (tn):", tn)
    print("False Positives (fp):", fp)
    print("False Negatives (fn):", fn)
    print("True Positives (tp):", tp)
    print("Accuracy:", accuracy)
    print("Equal Error Rate (EER): {:.2f}%\n".format(eer * 100))

    return {
        "EER": eer,
        "ACC": accuracy,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "TP": tp,
    }, th
