import data as data
import datautil as util
import numpy as np
from perceptronsimple import PerceptronSimple as ps
from perceptronmulti import PerceptronMulti as pm
import sys

class NullDevice():
    def write(self, s):
        pass

def networkANDOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.ANDOR.testset, testepsilon)

def networkAND(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.AND.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.AND.testset, testepsilon)

def networkOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.OR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.OR.testset, testepsilon)

def networkXOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.XOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.XOR.testset, testepsilon)

def networkBinaryOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(25, 5, lRate, epsilon, data.BinaryOCR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.SimpleOCR.testset, testepsilon)

def networkBipolarOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(25, 5, lRate, epsilon, data.BipolarOCR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_epoch)
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def networkMultiOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(25, [8,8], 5, lRate, epsilon, data.BipolarOCR.trainingset)
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
            networkANDOR(0.05,0.01,0.1)
        elif("and" in argv):
            networkAND(0.05,0.01,0.1)
        elif("or" in argv):
            networkOR(0.05,0.01,0.1)
        elif("xor" in argv):
            networkXOR(0.05,0.01,0.1)
        elif("bipolarocr" in argv):
            networkBipolarOCR(0.01,0.01,0.1)
        elif("multiocr" in argv):
            networkMultiOCR(0.07,0.01,0.1)
        else:
            a = ps(25, 5, 0.1, 0.05, data.BipolarOCR.trainingset)
            sys.stdout = original_stdout
            a.plotErrorThroughLearning(a.errors_in_each_epoch)
            a.testNetwork(data.BipolarOCR.testset, 0.1)
            a.testNetwork(util.getTestSetWithNoise(data.BipolarOCR.testset, 0.02), 0.1)

# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    main(sys.argv[1:])
