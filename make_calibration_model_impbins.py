#!/usr/bin/python
# make a predicted CTR to a calibration CTR
import sys
from sys import argv

def get_model(src_file, dest_file, impbinnum):
    '''src_file format: probability true_label'''
    fd = open(src_file)
    dfd = open(dest_file,"w")
    impbins = int(impbinnum)

    index = 0
    s_prob = 0
    s_clk = 0

    for line in fd:
        arr = line.rstrip().split('\t')
        prob = float(arr[0])
        label = arr[1]
        s_prob += prob
        s_clk += int(label)
        index += 1
        
        if index == impbins:
            avgctr = 1.0 * s_prob / index
            realctr = 1.0 * (s_clk+1) / (index+1000)
            print >> dfd, "\t".join([str(avgctr), str(realctr), "|".join([str(s_clk), str(index)])])
            index = 0
            s_prob = 0
            s_clk = 0

    fd.close()
    dfd.close()

if __name__ == "__main__":
    if len(argv) != 4:
        print "python %s srcfile impbinfile impbinsnum" % __file__
        sys.exit(0)
    get_model(argv[1],argv[2],argv[3])

