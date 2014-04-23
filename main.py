import data as data
import datautil as util
import numpy as np
from perceptronsimple import PerceptronSimple as ps
from perceptronmulti import PerceptronMulti as pm
import validacion as val
import sys

class NullDevice():
    def write(self, s):
        pass

def simpleANDOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.ANDOR.testset, testepsilon)

def simpleAND(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.AND.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.AND.testset, testepsilon)

def simpleOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.OR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.OR.testset, testepsilon)

def simpleXOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(2, 1, lRate, epsilon, data.XOR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.XOR.testset, testepsilon)

def simpleOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = ps(25, 5, lRate, epsilon, data.BipolarOCR.trainingset)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def multiANDOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [10], 2, lRate, epsilon)
    a.train_network(data.ANDOR.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.ANDOR.testset, testepsilon)

def multiAND(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [10], 1, lRate, epsilon)
    a.train_network(data.AND.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.AND.testset, testepsilon)

def multiOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [10], 1, lRate, epsilon)
    a.train_network(data.OR.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.OR.testset, testepsilon)

def multiXOR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [3], 1, lRate, epsilon)
    a.train_network(data.XOR.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.XOR.testset, testepsilon)
    
def multiOCR(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(25, [10], 5, lRate, epsilon)
    a.train_network(data.BipolarOCR.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def multiSin(lRate, epsilon, testepsilon):
    if(main.silent):
        sys.stdout = NullDevice()
    a = pm(2, [10], 1, lRate, epsilon)
    a.train_network(data.Sin.trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
    sys.stdout = main.original_stdout
    a.plotErrorThroughLearning()
    a.testNetwork(data.Sin.trainingset, testepsilon)

def main(argv):
    main.original_stdout = sys.stdout
    if(len(argv) > 0):
        if("-s" in argv):
            print "Shutting down debug output"
            print "Training network..."
            main.silent = True
        else:
            main.silent = False

        if("-validate" in argv):
            main.validacion = True
        else:
            main.validacion = False

        if("andor" in argv):
            simpleANDOR(0.05,0.01,0.1)
        elif("and" in argv):
            simpleAND(0.05,0.01,0.1)
        elif("or" in argv):
            simpleOR(0.05,0.01,0.1)
        elif("xor" in argv):
            simpleXOR(0.05,0.01,0.1)
        elif("ocr" in argv):
            simpleOCR(0.05,0.01,0.1)
            #a.testNetwork(util.getTestSetWithNoise(data.BipolarOCR.testset, 0.02), 0.1)
        elif("multiand" in argv):
            multiAND(0.05,0.01,0.1)
        elif("multior" in argv):
            multiOR(0.05,0.01,0.1)
        elif("multixor" in argv):
            multiXOR(0.2,0.01,0.1)
        elif("multiandor" in argv):
            multiANDOR(0.05,0.01,0.1)
        elif("multiocr" in argv):
            if(main.validacion):
                val.validateMultiOCR(plot = False)
            else:
                multiOCR(0.05,0.01,0.1)
        elif("multisin" in argv):
            if(main.validacion):
                val.validateMultiSin()
            else:
                multiSin(0.2,0.01,0.1)

# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    main(sys.argv[1:])
