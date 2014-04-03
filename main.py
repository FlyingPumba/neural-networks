import data as data
from perceptronsimple import PerceptronSimple as ps
import sys

class NullDevice():
    def write(self, s):
        pass

def networkANDOR(lRate, epsilon, testepsilon):
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.ANDOR.testset, testepsilon)

def networkAND(lRate, epsilon, testepsilon):
    a = ps(2, 1, lRate, epsilon, data.AND.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.AND.testset, testepsilon)

def networkOR(lRate, epsilon, testepsilon):
    a = ps(2, 1, lRate, epsilon, data.OR.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.OR.testset, testepsilon)

def networkXOR(lRate, epsilon, testepsilon):
    a = ps(2, 1, lRate, epsilon, data.XOR.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.XOR.testset, testepsilon)

def networkBinaryOCR(lRate, epsilon, testepsilon):
    a = ps(25, 5, lRate, epsilon, data.BinaryOCR.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.SimpleOCR.testset, testepsilon)

def networkBipolarOCR(lRate, epsilon, testepsilon):
    a = ps(25, 5, lRate, epsilon, data.BipolarOCR.trainingset)
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def main(argv):
    original_stdout = sys.stdout
    if(argv[0] == "-s"):
        print "Shutting down debug output"
        print "Training network..."
        sys.stdout = NullDevice()
        
    #a = ps(25, 5, 0.2, 0.1, data.SimpleOCR.trainingset)
    a = ps(25, 5, 0.2, 0.1, data.BipolarOCR.trainingset)
    sys.stdout = original_stdout
    a.plotErrorThroughLearning(a.errors_in_each_training)
    a.testNetwork(data.BipolarOCR.testset, 0.1)
    a.testNetwork(data.BipolarOCR.getTestSetWithNoise(0.02), 0.1)

if __name__ == "__main__":
    main(sys.argv[1:])
