import data as data
import datautil as util
import numpy as np
from perceptronsimple import PerceptronSimple as ps
from perceptronmulti import PerceptronMulti as pm
import sys

class NullDevice():
    def write(self, s):
        pass

def simpleANDOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.ANDOR.testset, testepsilon)

def simpleAND(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.AND.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.AND.testset, testepsilon)

def simpleOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.OR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.OR.testset, testepsilon)

def simpleXOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.XOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.XOR.testset, testepsilon)

def simpleOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(25, 5, lRate, epsilon, data.BipolarOCR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def multiANDOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [500], 2, lRate, epsilon, data.ANDOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.ANDOR.testset, testepsilon)

def multiAND(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [500], 1, lRate, epsilon, data.AND.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.AND.testset, testepsilon)

def multiOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [500], 1, lRate, epsilon, data.OR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.OR.testset, testepsilon)

def multiXOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [500], 1, lRate, epsilon, data.XOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.XOR.testset, testepsilon)

def multiOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(25, [500], 5, lRate, epsilon, data.BipolarOCR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def main(argv):
    main.original_stdout = sys.stdout
    if(len(argv) > 0):
        if("-s" in argv):
            print "Shutting down debug output"
            print "Training network..."
            main.silent = True
        else:
            main.silent = False
        if("andor" in argv):
            simpleANDOR(0.05,0.01,0.1)
        elif("and" in argv):
            simpleAND(0.05,0.01,0.1)
        elif("or" in argv):
            simpleOR(0.05,0.01,0.1)
        elif("xor" in argv):
            simpleXOR(0.05,0.01,0.1)
        elif("ocr" in argv):
            simpleOCR(0.01,0.01,0.1)
            #a.testNetwork(util.getTestSetWithNoise(data.BipolarOCR.testset, 0.02), 0.1)
        elif("multiand" in argv):
            multiAND(0.01,0.01,0.1)
        elif("multior" in argv):
            multiOR(0.01,0.01,0.1)
        elif("multixor" in argv):
            multiXOR(0.1,0.01,0.2)
        elif("multiandor" in argv):
            multiANDOR(0.01,0.01,0.1)
        elif("multiocr" in argv):
            multiOCR(0.1,0.01,0.1)

# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    main(sys.argv[1:])
