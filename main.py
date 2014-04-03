import data as data
from perceptronsimple import PerceptronSimple as ps

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

def main():
    #a = ps(25, 5, 0.2, 0.1, data.SimpleOCR.trainingset)
    a = ps(25, 5, 0.2, 0.1, data.BipolarOCR.trainingset)
    a.testNetwork(data.BipolarOCR.testset, 0.1)
    a.testNetwork(data.BipolarOCR.getTestSetWithNoise(0.02), 0.1)

if __name__ == "__main__":
    main()
