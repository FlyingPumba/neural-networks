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

def networkSimpleOCR(lRate, epsilon):
    a = ps(25, 5, lRate, epsilon, data.SimpleOCR.trainingset)
    a.testNetwork(data.SimpleOCR.testset)

def main():
    a = ps(25, 5, 0.2, 0.1, data.SimpleOCR.trainingset)
    a.testNetwork(data.SimpleOCR.testset)

if __name__ == "__main__":
    main()