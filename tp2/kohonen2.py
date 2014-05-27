import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patch
import random as rnd

class Region():
    """Clase que representa una region de puntos"""
    def __init__(self, minX, maxX, minY, maxY):
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

    def pertenece(self, x, y):
        return self.minX <= x <= self.maxX and self.minY <= y <= self.maxY

class NoSupervisedNetwork():
    """Mapas auto-organizados de Kohonen - Parte 2"""
    def __init__(self):
        self.nInput = 2
        self.nOutput = [10,10]
        self.etaAlpha = 0.1
        self.sigmaAlpha  = 0.2

        self.etaHistory = []
        self.sigmaHistory = []

    def trainNetwork(self, dataset, stochastic=True):
        W = np.random.uniform(-0.1,0.1,size=(self.nOutput[0], self.nInput, self.nOutput[1]))

        cant_epochs = 1
        max_epochs = 26

        while cant_epochs <= max_epochs:
            # begin a new epoch
            print "Epoch: %d, eta: %.8f, sigma: %.8f" % (cant_epochs, self.eta(cant_epochs), self.sigma(cant_epochs))

            self.etaHistory.append(self.eta(cant_epochs))
            self.sigmaHistory.append(self.sigma(cant_epochs))

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            for X in trainingset:
                Y = self.activation(X,W)
                P = self.winner(Y)
                D = self.proxy(P, self.sigma(cant_epochs))
                dW = []
                for i in xrange(len(W)):
                    aux = self.eta(cant_epochs) * (np.array([X]).T - W[i]) * D[i]
                    dW.append(aux)
                W = W + dW

            cant_epochs = cant_epochs + 1

        self.W = W

    def activation(self, X, W):
        Y = []
        for i in xrange(len(W)):
            Yaux = (W[i] - np.array([X]).T)**2
            Yaux = np.apply_along_axis(np.linalg.norm, 0, Yaux)
            Y.append(Yaux)

        Y = np.array(Y).flatten()
        return [True if x == min(Y) else False for x in Y]

    def eta(self, t):
        return t**(-self.etaAlpha)

    def sigma(self, t):
        return t**(-self.sigmaAlpha)

    def winner(self, Y):
        indice = Y.index(True)
        i = indice / self.nOutput[0]
        j = indice % self.nOutput[1]

        return [i,j]

    def proxy(self, p, sigma):
        d = np.zeros((self.nOutput[0],self.nOutput[1]))
        for i in xrange(self.nOutput[0]):
            for j in xrange(self.nOutput[1]):
                normaAlCuadrado = (p[0] - i)**2 + (p[1] - j)**2
                d[i,j] = np.exp(-normaAlCuadrado/(2*sigma**2))

        return d

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

    def rotular(self, dataset, regiones):
        act = np.zeros((self.nOutput[0], self.nOutput[1], len(regiones)))
        for X in dataset:
            Y = self.activation(X,self.W)
            p = self.winner(Y)

            # veo de que region es el punto
            for i in xrange(len(regiones)):
                if regiones[i].pertenece(X[0],X[1]):
                    act[p[0],p[1],i] += 1

        reg = np.zeros((self.nOutput[0], self.nOutput[1]))
        for i in xrange(self.nOutput[0]):
            for j in xrange(self.nOutput[1]):
                reg[i,j] =  np.where(act[i,j] == np.max(act[i,j]))[0][0]
                if not(np.all(act[i,j] == 0)):
                    reg[i,j] += 1

        self.regiones = reg

        colormap = np.array(['black','r', 'g', 'b', 'y'])
        points = []
        for i in xrange(self.nOutput[0]):
            for j in xrange(self.nOutput[1]):
                points.append([i,j])

        aux = zip(*points)

        # regiones en las neuronas
        #plt.subplot2grid((2,2), (0,0))
        plt.subplot(121)
        plt.xlabel("Mapa de Regiones")
        plt.scatter(aux[0], aux[1], s=100, c=colormap[reg.flatten().astype(np.int64)])

        # dataset
        #sp = plt.subplot2grid((2,2), (0,1))
        sp = plt.subplot(122)
        plt.xlabel("Dataset")
        datazipped = zip(*dataset)

        # regiones en el dataset
        if separados:
            sp.add_patch(patch.Rectangle((10,10),20,20, fill=False, color='r'))
            sp.add_patch(patch.Rectangle((40,10),20,20, fill=False, color='g'))
            sp.add_patch(patch.Rectangle((10,40),20,20, fill=False, color='b'))
            sp.add_patch(patch.Rectangle((40,40),20,20, fill=False, color='y'))
        else:
            sp.add_patch(patch.Rectangle((10,10),20,20, fill=False, color='r'))
            sp.add_patch(patch.Rectangle((30,10),20,20, fill=False, color='g'))
            sp.add_patch(patch.Rectangle((10,30),20,20, fill=False, color='b'))
            sp.add_patch(patch.Rectangle((30,30),20,20, fill=False, color='y'))        

        plt.scatter(datazipped[0], datazipped[1], s=10)

        # show eta and sigma history
        #sp2 = plt.subplot2grid((2,2), (1,0), colspan=2)
        #labelEta = 'eta (alpha: %.2f)' % self.etaAlpha
        #sp2.plot(self.etaHistory, label=labelEta)
        #labelSigma = 'sigma (alpha: %.2f)' % self.sigmaAlpha
        #sp2.plot(self.sigmaHistory, label=labelSigma)
        #sp2.legend(loc='upper right', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=3)

        plt.show()

    def validateNetwork(self, validationset, regiones):
        correcto = [0]*len(validationset)
        for i in xrange(len(validationset)):
            X = validationset[i]
            Y = self.activation(X,self.W)
            p = self.winner(Y)
            regionActivada = self.regiones[p[0],p[1]]

            # veo de que region es el punto
            for j in xrange(len(regiones)):
                if regiones[j].pertenece(X[0],X[1]) and j+1==regionActivada:
                    correcto[i] = 1

        plt.title("Validacion")
        colormap = np.array(['r', 'g'])
        datazipped = zip(*validationset)
        plt.scatter(datazipped[0], datazipped[1], s=50, c=colormap[correcto])

        p1 = patch.Rectangle((0, 0), 1, 1, fc="r")
        p2 = patch.Rectangle((0, 0), 1, 1, fc="g")
        labelWrong = 'Wrong points: %d' % correcto.count(0)
        labelRight = 'Right points: %d' % correcto.count(1)
        plt.legend((p1, p2), (labelWrong,labelRight), loc='upper center', ncol=3)

        plt.show()

# ========== DATASET ==========

def generateDataset(cant):
    dataset = []

    for i in xrange(cant):
        # choose a region
        r = np.random.randint(4)

        if r == 0:
            x = np.random.uniform(r1.minX,r1.maxX)
            y = np.random.uniform(r1.minY,r1.maxY)
            dataset.append(np.array([x,y]))
        elif r == 1:
            x = np.random.uniform(r2.minX,r2.maxX)
            y = np.random.uniform(r2.minY,r2.maxY)
            dataset.append(np.array([x,y]))
        elif r == 2:
            x = np.random.uniform(r3.minX,r3.maxX)
            y = np.random.uniform(r3.minY,r3.maxY)
            dataset.append(np.array([x,y]))
        elif r == 3:
            x = np.random.uniform(r4.minX,r4.maxX)
            y = np.random.uniform(r4.minY,r4.maxY)
            dataset.append(np.array([x,y]))

    return np.array(dataset)

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

separados = True
if separados:
    r1 = Region(10,30,10,30)
    r2 = Region(40,60,10,30)
    r3 = Region(10,30,40,60)
    r4 = Region(40,60,40,60)
else:
    r1 = Region(10,30,10,30)
    r2 = Region(30,50,10,30)
    r3 = Region(10,30,30,50)
    r4 = Region(30,50,30,50)

if __name__ == "__main__":
    net = NoSupervisedNetwork()

    # generate the data and validation sets
    cant_patterns_training = 400
    #cant_patterns_validation = 400

    data = generateDataset(cant_patterns_training)

    validation = generateDataset(cant_patterns_training)

    net.trainNetwork(data)
    print "Final weights: %s" % net.W

    net.rotular(data, [r1,r2,r3,r4])

    net.validateNetwork(validation, [r1,r2,r3,r4])

    #net.validateNetwork(normal_valset)
    #net.plotWeights()
