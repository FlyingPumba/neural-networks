import data as data
from perceptronsimple import PerceptronSimple as ps

def networkANDOR(lRate, epsilon):
    a = ps(2, 2, lRate, epsilon, data.ANDOR.trainingset)

    print "\nTesting the network"
    cant_patterns = data.ANDOR.testset.shape[0]
    for i in range(0,cant_patterns):
        a.evaluate(data.ANDOR.testset[i])

def networkAND(lRate, epsilon):
    a = ps(2, 2, lRate, epsilon, data.AND.trainingset)

    print "\nTesting the network"
    cant_patterns = data.AND.testset.shape[0]
    for i in range(0,cant_patterns):
        a.evaluate(data.AND.testset[i])

def networkOR(lRate, epsilon):
    a = ps(2, 2, lRate, epsilon, data.OR.trainingset)

    print "\nTesting the network"
    cant_patterns = data.OR.testset.shape[0]
    for i in range(0,cant_patterns):
        a.evaluate(data.OR.testset[i])

def networkXOR(lRate, epsilon):
    a = ps(2, 2, lRate, epsilon, data.XOR.trainingset)

    print "\nTesting the network"
    cant_patterns = data.XOR.testset.shape[0]
    for i in range(0,cant_patterns):
        a.evaluate(data.XOR.testset[i])