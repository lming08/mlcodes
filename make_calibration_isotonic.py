#!/usr/bin/python
# make calibration CTR with IsotonicRegression from sklearn

import sys
from sys import argv

from sklearn.isotonic import IsotonicRegression as IR
import numpy as np

def get_score_label(modelfile):
    score_list = []
    label_list = []
    fd = open(modelfile)
    for line in fd:
        arr = line.rstrip().split('\t')
        score = float(arr[0])
        label = int(arr[1])
        score_list.append(score)
        label_list.append(label)
    return score_list, label_list

def get_fit_model(score_list, label_list):
    p_train = np.array(score_list)
    y_train = np.array(label_list)

    ir = IR()
    ir.fit( p_train, y_train )
    return ir

def get_calibrate_res(ir, score_list):
    p_test = np.array(score_list)
    p_calibrated = ir.transform(p_test)
    return p_calibrated

def print_score_calibrated(score_list, p_calibrated, resfile):
    fd = open(resfile, "w")
    length = len(score_list)
    for index in range(length):
        score = score_list[index]
        p_calibr = p_calibrated[index]
        print >> fd, "\t".join([str(score), str(p_calibr)])

if __name__ == "__main__":
    if len(argv) != 3:
        print "python %s modelfile calibrfile" % __file__
        sys.exit(0)
    score_list, label_list = get_score_label(argv[1])
    ir = get_fit_model(score_list, label_list)
    p_calibrated = get_calibrate_res(ir, score_list)
    print_score_calibrated(score_list, p_calibrated, argv[2])

