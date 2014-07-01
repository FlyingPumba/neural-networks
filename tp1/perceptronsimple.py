import numpy as np
import matplotlib.pyplot as plt
import data as data
import datautil as du

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes, learning_rate, epsilon):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.lRate = learning_rate
        self.epsilon = epsilon

    def trainNetwork(self, dataset, batch=False, stochastic=True):
        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]
        trainingset = np.copy(dataset)

        # create the Weight matrix (nInput+1 for the threshold)
        W = np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nOutput))

        cant_epochs = 0
        max_epochs = 1000

        self.errors_in_each_epoch = []
        self.appendEpochError = self.errors_in_each_epoch.append

        self.errors_in_each_epoch_sum = []
        self.appendEpochErrorSum = self.errors_in_each_epoch_sum.append

        while True:
            # begin a new epoch
            # to store the square error
            errors_in_each_pattern = []
            appendPatternError = errors_in_each_pattern.append

            errors_in_each_pattern_abs = []
            appendPatternErrorAbs = errors_in_each_pattern_abs.append

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            if(batch):
                D = np.zeros((self.nInput+1, self.nOutput))

            for i in xrange(cant_patterns):
                X = self.getInputWithThreshold(trainingset[i,0])
                Y = np.tanh(np.dot(X,W))
                Z = trainingset[i,1]

                # calculate the error
                E = Z - Y
                
                appendPatternError(np.dot(E, E))
                appendPatternErrorAbs(np.sum(np.absolute(E)/2))

                # calculate the delta
                transposedX = np.array([X]).T
                delta = self.lRate * transposedX * E
                
                # learn !
                if(batch):
                    D = D + delta
                else:
                    W = W + delta
            
            if(batch):
                W = W + D

            cant_epochs = cant_epochs + 1
            self.appendEpochError(np.max(errors_in_each_pattern))
            self.appendEpochErrorSum(np.sum(errors_in_each_pattern_abs))
            
            keep_going = True
            if(cant_epochs >= max_epochs):
                print "\nREACHED MAX EPOCHS\n"
                keep_going = False
            if(self.epsilon >= np.max(errors_in_each_pattern)):
                print "\nREACHED BETTER ERROR THAN EPSILON IN LAST EPOCH\n"
                keep_going = False

            if(keep_going == False):
                self.W = W
                print "total epochs = %d\n" % cant_epochs
                print "last e = %.10f\n" % np.max(errors_in_each_pattern)
                break

    def plotErrorThroughLearning(self):
        answer = ""
        while (answer != "y") & (answer != "n"):
            answer = raw_input("Do you wanna see the error through learning ? (y/n) ")

        if(answer == "y"):
            plt.plot(self.errors_in_each_epoch)
            plt.plot(self.errors_in_each_epoch_sum)
            plt.ylabel("network error")
            plt.xlabel("epoch number")
            plt.show()

    def getInputWithThreshold(self, input):
        # create input array
        X = np.zeros(self.nInput+1)
        for j in xrange(self.nInput):
            X[j] = input[j]
        X[self.nInput] = -1

        return X

    def testNetwork(self, testset, testepsilon):
        cant_patterns = testset.shape[0]
        patterns_with_error = 0

        for i in xrange(cant_patterns):
            print "\nTesting pattern %d" % (i+1)
            X = self.getInputWithThreshold(testset[i,0])
            Y = np.tanh(np.dot(X,self.W))
            print "Output is: %s" % Y

            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = Z - Y
            print "Error is: %s" % E

            absolute_error = np.absolute(E)/2
            sum_error = np.sum(absolute_error)
            if(sum_error>testepsilon):
                print "-----> WRONG OUTPUT. The sum error for this pattern is: %.5f" % sum_error
                patterns_with_error = patterns_with_error + 1

        print "\nThere were %d errors over %d patterns" % (patterns_with_error, cant_patterns)
        
# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    a = PerceptronSimple(25, 5, 0.05, 0.01)
    a.trainNetwork(data.BipolarOCR.trainingset)
    a.plotErrorThroughLearning()
    
    # ========== VALIDATION WITH ORIGINAL LETTERS ==========
    print "\n VALIDATION with original letters\n"
    a.testNetwork(data.BipolarOCR.trainingset, 0.1)
    raw_input()
    
    # ========== VALIDATION WITH MODIFIED LETTERS ==========

    testSet05 = []
    testSet10 = []
    testSet15 = []
    testSet20 = []
    testSet30 = []
    testSet40 = []
    testSet50 = []
    
    for i in xrange(len(data.BipolarOCR.trainingset)):
        #du.plotLetter(data.BipolarOCR.trainingset[i,0], saveFile=True)
        pattern05 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.05)       
        testSet05.append(np.array([pattern05, data.BipolarOCR.trainingset[i,1]]))

    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern10 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.1)
        testSet.append(np.array([pattern10, data.BipolarOCR.trainingset[i,1]]))
        
    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern15 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.15)
        testSet.append(np.array([pattern15, data.BipolarOCR.trainingset[i,1]]))
        
    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern20 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.2)
        testSet.append(np.array([pattern20, data.BipolarOCR.trainingset[i,1]]))
        
    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern30 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.3)
        testSet.append(np.array([pattern30, data.BipolarOCR.trainingset[i,1]]))

    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern40 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.4)
        testSet.append(np.array([pattern40, data.BipolarOCR.trainingset[i,1]]))
        
    for i in xrange(len(data.BipolarOCR.trainingset)):
        pattern50 = du.getPatternWithNoise(data.BipolarOCR.trainingset[i,0], 0.5)
        testSet.append(np.array([pattern50, data.BipolarOCR.trainingset[i,1]]))

    print "\n VALIDATION with modified letters - noise rate: 05%\n"
    a.testNetwork(np.asarray(testSet05), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 10%\n"
    a.testNetwork(np.asarray(testSet10), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 15%\n"
    a.testNetwork(np.asarray(testSet15), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 20%\n"
    a.testNetwork(np.asarray(testSet20), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 30%\n"
    a.testNetwork(np.asarray(testSet30), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 40%\n"
    a.testNetwork(np.asarray(testSet40), 0.1)
    raw_input()
    print "\n VALIDATION with modified letters - noise rate: 50%\n"
    a.testNetwork(np.asarray(testSet50), 0.1)

