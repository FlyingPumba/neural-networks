import data as data
from perceptronsimple import PerceptronSimple as ps
import sys

class NullDevice():
    def write(self, s):
        pass

def networkANDOR(lRate, epsilon):
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)
    a.testNetwork(data.ANDOR.testset)

def networkAND(lRate, epsilon):
    a = ps(2, 1, lRate, epsilon, data.AND.trainingset)
    a.testNetwork(data.AND.testset)

def networkOR(lRate, epsilon):
    a = ps(2, 1, lRate, epsilon, data.OR.trainingset)
    a.testNetwork(data.OR.testset)

def networkXOR(lRate, epsilon):
    a = ps(2, 1, lRate, epsilon, data.XOR.trainingset)
    a.testNetwork(data.XOR.testset)

def networkBinaryOCR(lRate, epsilon, testepsilon):
    a = ps(25, 5, lRate, epsilon, data.BinaryOCR.trainingset)
    a.testNetwork(data.SimpleOCR.testset, testepsilon)

def networkBipolarOCR(lRate, epsilon, testepsilon):
    a = ps(25, 5, lRate, epsilon, data.BipolarOCR.trainingset)
    a.testNetwork(data.BipolarOCR.testset, testepsilon)

def main(argv):
    original_stdout = sys.stdout
    if(argv[0] == "-s"):
        print "Shutting down debug output"
        print "Training network..."
        sys.stdout = NullDevice()
        
    #a = ps(25, 5, 0.2, 0.1, data.SimpleOCR.trainingset)
    a = ps(25, 5, 0.2, 0.1, data.BipolarOCR.trainingset)
    print "debug text"
    sys.stdout = original_stdout
    a.testNetwork(data.BipolarOCR.testset, 0.1)
    a.testNetwork(data.BipolarOCR.getTestSetWithNoise(0.02), 0.1)
    print "test text"

if __name__ == "__main__":
    main(sys.argv[1:])
