import transformers
from sklearn import preprocessing
from sklearn.metrics import  classification_report
import time
import math
import torch
import numpy as np


# Initialize default tags


def_tags = ['1_i', '1_o', '1_p', 'N', 'X'] ##EBM x->[4]
labels = ['Intervation', 'Outcome', 'Patient', 'None'] ##EBM

enc_trans = preprocessing.LabelEncoder()
enc_trans = enc_trans.fit(def_tags)


def timeSince(since):
    """Transform time to min-sec"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def metrics(y_true, y_pred, labels=None, target_names=None):
    y_labels, y_labels_pre = [], []
    for l in y_true:
        y_labels.extend(l)

    for l in y_pred:
        y_labels_pre.extend(l)

    clf = classification_report(y_labels,y_labels_pre, labels=labels, target_names=target_names)
    print("\n=========================================\n")
    print('Classification Report:\n')
    print(clf)
    print("\n=========================================\n")

    return clf