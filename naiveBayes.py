#!/usr/bin/python
#  NaiveBayes(with M-smoothing) Feature Ranking through ROC AUC
#  Hairen Liao 2014.1.23
import sys
from sys import argv
import math
import numpy as np
from sklearn import metrics

# the format of src log file
[reqtime,fisrtid,userid,platform,host,geoip,region,city,browsegrp,browser, \
    osgrp,os,adid,orderid,campaignid,strgyid,creatid,adunitid,width,height, \
    location,viewtype,type,floorprice,bidprice,winprice,clkcnt,reachcnt,convcnt, \
    usernew,agentprod,deviceos] = range(0, 32)

class NaiveBayes:
    def __init__(self, trainFile):
        self.trainModel(trainFile)
        self.M0 = 0.
        self.C0 = 0.

    def trainModel(self, trainModelFile):
        trainfd = open(trainModelFile)
        self.model = {}
        self.totalInfo = {}

        for line in trainfd:
            arr = line.rstrip().split("\t")
            pltf = arr[0]
            dimType = arr[1]
            dimName = arr[2]
            imp = int(arr[3])
            clk = int(arr[4])

            key = "|".join([pltf, dimType, dimName])
            if key not in self.model:
                self.model[key] = [0, 0]
                self.model[key][0] += imp - clk
                self.model[key][1] += clk

            if pltf not in self.totalInfo:
                self.totalInfo[pltf] = [0, 0]
            self.totalInfo[pltf][0] += imp - clk
            self.totalInfo[pltf][1] += clk
        trainfd.close()


    def getAUC(self):
        y = np.array(self.testingY)
        pred = np.array(self.predictResult)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def predict(self, testFile):
        self.testingData = []
        #self.readData(testFile, self.testingData)
        self.testingY = []
        self.predictResult = []

        fd = open(testFile)
        #for instance in self.testingData:
        for line in fd:
           arr = line.rstrip().split('\t')
           clickcount = int(arr[clkcnt])
           if clickcount > 0:
               clickcount = 1

           f = 0.
           pltf = arr[platform]
           Domain = arr[host]
           Region = arr[region]
           City = arr[city]
           Browser = arr[browser]
           Os = arr[os]
           Creatid = arr[creatid]
           Size = arr[width]+"*"+arr[height]
           Location = arr[location]
           AdslotId = arr[adunitid]

           priorRatio = 1.0 * self.totalInfo[pltf][1] / (self.totalInfo[pltf][0])
           priorProb = 1.0 * self.totalInfo[pltf][1] / (self.totalInfo[pltf][1] + self.totalInfo[pltf][0])
           C0 = priorProb * self.M0

           # domain  
           f += math.log( (self.getPosNum(pltf,"domain",Domain) + C0) / ((self.getNegNum(pltf,"domain",Domain) + self.M0 - C0) * priorRatio) )
           # region
           f += math.log( (self.getPosNum(pltf,"region",Region) + C0) / ((self.getNegNum(pltf,"region",Region) + self.M0 - C0) * priorRatio) )
           # city
           f += math.log( (self.getPosNum(pltf,"city",City) + C0) / ((self.getNegNum(pltf,"city",City) + self.M0 - C0) * priorRatio) )
           # browser
           f += math.log( (self.getPosNum(pltf,"browser",Browser) + C0) / ((self.getNegNum(pltf,"browser",Browser) + self.M0 - C0) * priorRatio) )
           # os 
           f += math.log( (self.getPosNum(pltf,"os",Os) + C0) / ((self.getNegNum(pltf,"os",Os) + self.M0 - C0) * priorRatio) )
           # creatid
           f += math.log( (self.getPosNum(pltf,"creatid",Creatid) + C0) / ((self.getNegNum(pltf,"creatid",Creatid) + self.M0 - C0) * priorRatio) )
           # size
           f += math.log( (self.getPosNum(pltf,"size",Size) + C0) / ((self.getNegNum(pltf,"size",Size) + self.M0 - C0) * priorRatio) )

           # location
           f += math.log( (self.getPosNum(pltf,"location",Location) + C0) / ((self.getNegNum(pltf,"location",Location) + self.M0 - C0) * priorRatio) )

           # adslotid
           f += math.log( (self.getPosNum(pltf,"adslotid",AdslotId) + C0) / ((self.getNegNum(pltf,"adslotid",AdslotId) + self.M0 - C0) * priorRatio) )

           posteriorRatio = priorRatio * math.exp(f)
           # posterior probability
           prob = posteriorRatio / (1. + posteriorRatio)

           self.predictResult.append(prob)
           self.testingY.append(clickcount)

    def getPosNum(self, platform, dimType, dimName):
        key = "|".join([platform, dimType, dimName])
        if key not in self.model:
            return 0
        num = self.model[key][1]
        return num

    def getNegNum(self, platform, dimType, dimName):
        key = "|".join([platform, dimType, dimName])
        if key not in self.model:
            return 0
        num = self.model[key][0]
        return num

    def setMestimateSmoothingPara(self, m):
        self.M0 = m

    def savePredictResult(self, resultFile):
        f = open(resultFile, 'w')
        for prob in self.predictResult:
            print >>f, prob

    def savePredictResultTrueClass(self, resultFile):
        f = open(resultFile, 'w')
        for index in range(len(self.predictResult)):
            prob = self.predictResult[index]
            true_class = self.testingY[index]
            print >>f, "\t".join([str(prob), str(true_class)])


def main(trainFile, testFile):
    nb = NaiveBayes(trainFile)
    nb.setMestimateSmoothingPara(300)
    # nb.setLaplaceSmoothing()
    nb.predict(testFile)
    print("after training %s,the value of AUC: %f" % (trainFile, nb.getAUC()))
    nb.savePredictResultTrueClass(testFile+"_adslot_predProb_trueClass.txt")

if __name__ == "__main__":
    if len(argv) != 3:
        print "Usage: python %s trainFile(in) testFile(out)" % __file__
        sys.exit(0)

    main(argv[1], argv[2])

