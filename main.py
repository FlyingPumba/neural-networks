import data as data
from perceptronsimple import PerceptronSimple as ps

def main():
    
    # input nodes: 2, output nodes: 2, lRate: 0.2, epsilon: 0.1
    a = ps(2, 2, 0.2, 0.1, data.ANDOR.trainingset)

    print "\nTesting the network"
    for i in range(0,4):
        a.evaluate(data.ANDOR.testset[i])

    # print some nice graphics
    #plt.plot(testset)
    #plt.ylabel("test set")
    #plt.show()


if __name__ == "__main__":
    main()