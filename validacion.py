import data as data
import datautil as util
import numpy as np
from perceptronmulti import PerceptronMulti as pm

# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

def validateMultiOCR(plot = False):
    net = pm(25, [10], 5, 0.05, 0.01)
    net.train_network(data.BipolarOCR.trainingset, momentum=False, stochastic=True)
    net.plotErrorThroughLearning()

    validationset1 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.02)
    validationset2 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.05)
    validationset3 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.1)
    validationset4 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.3)
    validationset5 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.5)
    validationset6 = util.getTestSetWithNoise(data.BipolarOCR.testset, 0.8)
    validationset7 = util.getTestSetWithNoise(data.BipolarOCR.testset, 1.5)

    print "Errors with 2%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset1, 0.1)
    raw_input()
    print "Errors with 5%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset2, 0.1)
    raw_input()
    print "Errors with 10%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset3, 0.1)
    raw_input()
    print "Errors with 30%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset4, 0.1)
    raw_input()
    print "Errors with 50%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset5, 0.1)
    raw_input()
    print "Errors with 80%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset6, 0.1)
    raw_input()
    print "Errors with 150%% of noise (testepsilon: 0.1): %s" % net.testNetwork(validationset7, 0.1)
    raw_input()

    validationset5 = util.getTestSetWithSwitchedUnits(data.BipolarOCR.testset, 0.02)
    validationset6 = util.getTestSetWithSwitchedUnits(data.BipolarOCR.testset, 0.05)
    validationset7 = util.getTestSetWithSwitchedUnits(data.BipolarOCR.testset, 0.1)
    validationset8 = util.getTestSetWithSwitchedUnits(data.BipolarOCR.testset, 0.3)

    print "Errors with 2%% of swithced units (testepsilon: 0.1): %s" % net.testNetwork(validationset5, 0.1)
    raw_input()
    print "Errors with 5%% of swithced units (testepsilon: 0.1): %s" % net.testNetwork(validationset6, 0.1)
    raw_input()
    print "Errors with 10%% of swithced units (testepsilon: 0.1): %s" % net.testNetwork(validationset7, 0.1)
    raw_input()
    print "Errors with 30%% of swithced units (testepsilon: 0.1): %s" % net.testNetwork(validationset8, 0.1)
    raw_input()

def validateMultiSin(plot = False):
    net1 = pm(2, [20], 1, 0.2, 0.01)
    net2 = pm(2, [20], 1, 0.2, 0.01)
    #net3 = pm(2, [20], 1, 0.2, 0.01)
    #net4 = pm(2, [20], 1, 0.2, 0.01)

    net1.train_network(data.Sin.trainingset, stochastic=True)
    net2.train_network(data.Sin.trainingset, momentum=True, stochastic=True)
    #net3.train_network(trainingset, dlr=True, stochastic=True)
    #net4.train_network(trainingset, momentum=True, dlr=True, stochastic=True)

    if plot:
        net1.plotErrorThroughLearning()
        net2.plotErrorThroughLearning()
        #net3.plotErrorThroughLearning()
        #net4.plotErrorThroughLearning()

    print "\nErrors for interpolationset. Net: not mods. Errors: %d/%d" % (net1.testNetwork(data.Sin.interpolationset, 0.1), len(data.Sin.interpolationset))
    print "Errors for interpolationset. Net: momentum. Errors: %d/%d" % (net2.testNetwork(data.Sin.interpolationset, 0.1), len(data.Sin.interpolationset))

    print "\nErrors for extrapolationset. Net: not mods. Errors: %d/%d" % (net1.testNetwork(data.Sin.extrapolationset, 0.1), len(data.Sin.extrapolationset))
    print "Errors for extrapolationset. Net: momentum. Errors: %d/%d" % (net2.testNetwork(data.Sin.extrapolationset, 0.1), len(data.Sin.extrapolationset))

def validateMultiOCRALL(plot = False):
    # split the training set into 10 arrays
    nChunks = 4
    chunks = np.array_split(data.BipolarOCR.trainingset, nChunks)

    a1CantErrorsOnTest = []
    a2CantErrorsOnTest = []
    a3CantErrorsOnTest = []
    b1CantErrorsOnTest = []
    b2CantErrorsOnTest = []
    b3CantErrorsOnTest = []

    c1CantErrorsOnTest = []
    c2CantErrorsOnTest = []
    c3CantErrorsOnTest = []
    d1CantErrorsOnTest = []
    d2CantErrorsOnTest = []
    d3CantErrorsOnTest = []

    e1CantErrorsOnTest = []
    e2CantErrorsOnTest = []
    e3CantErrorsOnTest = []
    f1CantErrorsOnTest = []
    f2CantErrorsOnTest = []
    f3CantErrorsOnTest = []

    g1CantErrorsOnTest = []
    g2CantErrorsOnTest = []
    g3CantErrorsOnTest = []
    h1CantErrorsOnTest = []
    h2CantErrorsOnTest = []
    h3CantErrorsOnTest = []


    for i in xrange(nChunks):
        # the i-th chunk is going to be the validation array
        validationSet = np.array(chunks[i])
        trainingset = []
        # add al the other chunks to the trainingset
        for j in xrange(nChunks):
            if j != i:
                for k in xrange(len(chunks[j])):
                    trainingset.append(np.array(chunks[j][k]))

        trainingset = np.array(trainingset)

        print "validationSet: %s" % validationSet
        raw_input()

        # test this training set with different parameters

        # pm(nInput, nHidden, nOutput, lRate, epsilon)
        a1 = pm(2, [10], 1, 0.05, 0.01)
        a2 = pm(2, [10], 1, 0.1, 0.01)
        a3 = pm(2, [10], 1, 0.2, 0.01)

        b1 = pm(2, [50], 1, 0.05, 0.01)
        b2 = pm(2, [50], 1, 0.1, 0.01)
        b3 = pm(2, [50], 1, 0.2, 0.01)

        c1 = pm(2, [10], 1, 0.05, 0.01)
        c2 = pm(2, [10], 1, 0.1, 0.01)
        c3 = pm(2, [10], 1, 0.2, 0.01)

        d1 = pm(2, [50], 1, 0.05, 0.01)
        d2 = pm(2, [50], 1, 0.1, 0.01)
        d3 = pm(2, [50], 1, 0.2, 0.01)

        e1 = pm(2, [10], 1, 0.05, 0.01)
        e2 = pm(2, [10], 1, 0.1, 0.01)
        e3 = pm(2, [10], 1, 0.2, 0.01)

        f1 = pm(2, [50], 1, 0.05, 0.01)
        f2 = pm(2, [50], 1, 0.1, 0.01)
        f3 = pm(2, [50], 1, 0.2, 0.01)

        g1 = pm(2, [10], 1, 0.05, 0.01)
        g2 = pm(2, [10], 1, 0.1, 0.01)
        g3 = pm(2, [10], 1, 0.2, 0.01)

        h1 = pm(2, [50], 1, 0.05, 0.01)
        h2 = pm(2, [50], 1, 0.1, 0.01)
        h3 = pm(2, [50], 1, 0.2, 0.01)

        a1.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        a2.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        a3.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        b1.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        b2.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        b3.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)

        c1.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)
        c2.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)
        c3.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)
        d1.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)
        d2.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)
        d3.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=False)

        e1.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        e2.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        e3.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        f1.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        f2.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)
        f3.train_network(trainingset, batch=False, stochastic=True, momentum=False, dlr=False)

        g1.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)
        g2.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)
        g3.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)
        h1.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)
        h2.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)
        h3.train_network(trainingset, batch=False, stochastic=True, momentum=True, dlr=True)

        if plot:
            print "a1: lRate: 0.05, epsilon: 0.01, nHidden: 10. No modifications"
            a1.plotErrorThroughLearning()
            print "a2: lRate: 0.1, epsilon: 0.01, nHidden: 10. No modifications"
            a2.plotErrorThroughLearning()
            print "a3: lRate: 0.2, epsilon: 0.01, nHidden: 10. No modifications"
            a3.plotErrorThroughLearning()
            print "b1: lRate: 0.05, epsilon: 0.01, nHidden: 50. No modifications"
            b1.plotErrorThroughLearning()
            print "b2: lRate: 0.1, epsilon: 0.01, nHidden: 50. No modifications"
            b2.plotErrorThroughLearning()
            print "b3: lRate: 0.2, epsilon: 0.01, nHidden: 50. No modifications"
            b3.plotErrorThroughLearning()

            print "c1: lRate: 0.05, epsilon: 0.01, nHidden: 10. Momentum"
            c1.plotErrorThroughLearning()
            print "c2: lRate: 0.1, epsilon: 0.01, nHidden: 10. Momentum"
            c2.plotErrorThroughLearning()
            print "c3: lRate: 0.2, epsilon: 0.01, nHidden: 10. Momentum"
            c3.plotErrorThroughLearning()
            print "d1: lRate: 0.05, epsilon: 0.01, nHidden: 50. Momentum"
            d1.plotErrorThroughLearning()
            print "d2: lRate: 0.1, epsilon: 0.01, nHidden: 50. Momentum"
            d2.plotErrorThroughLearning()
            print "d3: lRate: 0.2, epsilon: 0.01, nHidden: 50. Momentum"
            d3.plotErrorThroughLearning()

            print "e1: lRate: 0.05, epsilon: 0.01, nHidden: 10. DLR"
            e1.plotErrorThroughLearning()
            print "e2: lRate: 0.1, epsilon: 0.01, nHidden: 10. DLR"
            e2.plotErrorThroughLearning()
            print "e3: lRate: 0.2, epsilon: 0.01, nHidden: 10. DLR"
            e3.plotErrorThroughLearning()
            print "f1: lRate: 0.05, epsilon: 0.01, nHidden: 50. DLR"
            f1.plotErrorThroughLearning()
            print "f2: lRate: 0.1, epsilon: 0.01, nHidden: 50. DLR"
            f2.plotErrorThroughLearning()
            print "f3: lRate: 0.2, epsilon: 0.01, nHidden: 50. DLR"
            f3.plotErrorThroughLearning()

            print "g1: lRate: 0.05, epsilon: 0.01, nHidden: 10. Momentum & DLR"
            g1.plotErrorThroughLearning()
            print "g2: lRate: 0.1, epsilon: 0.01, nHidden: 10. Momentum & DLR"
            g2.plotErrorThroughLearning()
            print "g3: lRate: 0.2, epsilon: 0.01, nHidden: 10. Momentum & DLR"
            g3.plotErrorThroughLearning()
            print "h1: lRate: 0.05, epsilon: 0.01, nHidden: 50. Momentum & DLR"
            h1.plotErrorThroughLearning()
            print "h2: lRate: 0.1, epsilon: 0.01, nHidden: 50. Momentum & DLR"
            h2.plotErrorThroughLearning()
            print "h3: lRate: 0.2, epsilon: 0.01, nHidden: 50. Momentum & DLR"
            h3.plotErrorThroughLearning()

        a1CantErrorsOnTest.append(a1.testNetwork(validationSet, 0.1))
        a2CantErrorsOnTest.append(a2.testNetwork(validationSet, 0.1))
        a3CantErrorsOnTest.append(a3.testNetwork(validationSet, 0.1))
        b1CantErrorsOnTest.append(b1.testNetwork(validationSet, 0.1))
        b2CantErrorsOnTest.append(b2.testNetwork(validationSet, 0.1))
        b3CantErrorsOnTest.append(b3.testNetwork(validationSet, 0.1))

        c1CantErrorsOnTest.append(c1.testNetwork(validationSet, 0.1))
        c2CantErrorsOnTest.append(c2.testNetwork(validationSet, 0.1))
        c3CantErrorsOnTest.append(c3.testNetwork(validationSet, 0.1))
        d1CantErrorsOnTest.append(d1.testNetwork(validationSet, 0.1))
        d2CantErrorsOnTest.append(d2.testNetwork(validationSet, 0.1))
        d3CantErrorsOnTest.append(d3.testNetwork(validationSet, 0.1))

        e1CantErrorsOnTest.append(e1.testNetwork(validationSet, 0.1))
        e2CantErrorsOnTest.append(e2.testNetwork(validationSet, 0.1))
        e3CantErrorsOnTest.append(e3.testNetwork(validationSet, 0.1))
        f1CantErrorsOnTest.append(f1.testNetwork(validationSet, 0.1))
        f2CantErrorsOnTest.append(f2.testNetwork(validationSet, 0.1))
        f3CantErrorsOnTest.append(f2.testNetwork(validationSet, 0.1))

        g1CantErrorsOnTest.append(g1.testNetwork(validationSet, 0.1))
        g2CantErrorsOnTest.append(g2.testNetwork(validationSet, 0.1))
        g3CantErrorsOnTest.append(g2.testNetwork(validationSet, 0.1))
        h1CantErrorsOnTest.append(h1.testNetwork(validationSet, 0.1))
        h2CantErrorsOnTest.append(h2.testNetwork(validationSet, 0.1))
        h3CantErrorsOnTest.append(h3.testNetwork(validationSet, 0.1))

    print "Errors for a1: %s" % a1CantErrorsOnTest
    print "Errors for a2: %s" % a2CantErrorsOnTest
    print "Errors for a3: %s" % a3CantErrorsOnTest
    print "Errors for b1: %s" % b1CantErrorsOnTest
    print "Errors for b2: %s" % b2CantErrorsOnTest
    print "Errors for b3: %s" % b3CantErrorsOnTest

    print "Errors for c1: %s" % c1CantErrorsOnTest
    print "Errors for c2: %s" % c2CantErrorsOnTest
    print "Errors for c3: %s" % c3CantErrorsOnTest
    print "Errors for d1: %s" % d1CantErrorsOnTest
    print "Errors for d2: %s" % d2CantErrorsOnTest
    print "Errors for d3: %s" % d3CantErrorsOnTest

    print "Errors for e1: %s" % e1CantErrorsOnTest
    print "Errors for e2: %s" % e2CantErrorsOnTest
    print "Errors for e3: %s" % e3CantErrorsOnTest
    print "Errors for f1: %s" % f1CantErrorsOnTest
    print "Errors for f2: %s" % f2CantErrorsOnTest
    print "Errors for f3: %s" % f3CantErrorsOnTest

    print "Errors for g1: %s" % g1CantErrorsOnTest
    print "Errors for g2: %s" % g2CantErrorsOnTest
    print "Errors for g3: %s" % g3CantErrorsOnTest
    print "Errors for h1: %s" % h1CantErrorsOnTest
    print "Errors for h2: %s" % h2CantErrorsOnTest
    print "Errors for h3: %s" % h3CantErrorsOnTest