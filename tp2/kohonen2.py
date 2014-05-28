import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patch
import random as rnd

class Region():
    """Region de puntos"""
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
        self.etaAlpha = 0.2
        self.sigmaAlpha  = 0.1

        self.etaHistory = []
        self.sigmaHistory = []

    def trainNetwork(self, dataset, valset, stochastic=True):
        self.W = np.random.uniform(-0.1,0.1,size=(self.nOutput[0], self.nInput, self.nOutput[1]))

        cant_epochs = 1
        max_epochs = 20

        while cant_epochs <= max_epochs:
            # begin a new epoch
            #print "Epoch: %d, eta: %.8f, sigma: %.8f" % (cant_epochs, self.eta(cant_epochs), self.sigma(cant_epochs))

            self.etaHistory.append(self.eta(cant_epochs))
            self.sigmaHistory.append(self.sigma(cant_epochs))

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            for X in trainingset:
                Y = self.activation(X,self.W)
                P = self.winner(Y)
                D = self.proxy(P, self.sigma(cant_epochs))
                #print D
                dW = []
                for i in xrange(len(self.W)):
                    aux = self.eta(cant_epochs) * (np.array([X]).T - self.W[i]) * D[i]
                    dW.append(aux)
                self.W = self.W + dW

            print "Epoch: %d, eta: %.8f, sigma: %.8f" % (cant_epochs, self.eta(cant_epochs), self.sigma(cant_epochs))
            #self.rotularYValidar(dataset,valset, [r1,r2,r3,r4])
            #raw_input()
            cant_epochs = cant_epochs + 1

    def activation(self, X, W):
        Y = []
        for i in xrange(len(W)):
            Yaux = (W[i] - np.array([X]).T)**2
            Yaux = np.apply_along_axis(np.linalg.norm, 0, Yaux)
            Y.append(Yaux)

        Y = np.array(Y).flatten()
        return [True if x == min(Y) else False for x in Y]

    def eta(self, t):
        return self.etaAlpha

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
        plt.title("Final weights")
        points = self.W.reshape(self.nOutput[0],self.nOutput[1],self.nInput)
        auxX = []
        auxY = []
        for i in xrange(self.nOutput[0]):
            for j in xrange(self.nOutput[1]):
                auxX.append(points[i,j,0])
                auxY.append(points[i,j,1])

        sp =plt.subplot(111)
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

        plt.scatter(auxX, auxY, s=50)
        fileName = "kohonen2-weights-%d" % filekey
        plt.savefig(fileName, bbox_inches='tight')
        plt.show()

    def rotularYValidar(self, dataset, validationset, regiones):
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
        plt.title("Mapa de Regiones")
        scat = plt.scatter(aux[0], aux[1], s=100, c=colormap[reg.flatten().astype(np.int64)])

        # dataset
        #sp = plt.subplot2grid((2,2), (0,1))
        sp = plt.subplot(122)
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

        colormap = np.array(['r', 'g'])
        datazipped = zip(*validationset)
        plt.scatter(datazipped[0], datazipped[1], s=50, c=colormap[correcto])

        p1 = patch.Rectangle((0, 0), 1, 1, fc="r")
        p2 = patch.Rectangle((0, 0), 1, 1, fc="g")
        labelWrong = 'Wrong points: %d' % correcto.count(0)
        labelRight = 'Right points: %d' % correcto.count(1)
        plt.legend((p1, p2), (labelWrong,labelRight), loc='upper center', fancybox=True)

        supt = "Parametros: eta: %.2f sigma-alpha: %.2f" % (self.etaAlpha, self.sigmaAlpha)
        plt.suptitle(supt)
        fileName = 'kohonen2-%d.png' % filekey
        print fileName
        figure = plt.gcf() # get current figure
        figure.set_size_inches(10, 8) #this will give us a 800x600 image
        # when saving, specify the DPI
        plt.savefig(fileName, bbox_inches='tight', dpi = 100)
        plt.show()

# ========== DATASET ==========

def generateDataset(cant, plot=False, mismaDensidad=True):
    dataset = []

    for i in xrange(cant):
        if i%4 == 0:
            x = np.random.uniform(r1.minX,r1.maxX)
            y = np.random.uniform(r1.minY,r1.maxY)
            dataset.append(np.array([x,y]))
        elif i%4 == 1:
            x = np.random.uniform(r2.minX,r2.maxX)
            y = np.random.uniform(r2.minY,r2.maxY)
            dataset.append(np.array([x,y]))
        elif i%4 == 2:
            x = np.random.uniform(r3.minX,r3.maxX)
            y = np.random.uniform(r3.minY,r3.maxY)
            dataset.append(np.array([x,y]))
        elif i%4 == 3:
            x = np.random.uniform(r4.minX,r4.maxX)
            y = np.random.uniform(r4.minY,r4.maxY)
            dataset.append(np.array([x,y]))

    if not mismaDensidad:
        regionEspecial = np.random.randint(4)
        if regionEspecial%4 == 0:
            r = r1
        elif regionEspecial%4 == 1:
            r = r2
        elif regionEspecial%4 == 2:
            r = r3
        elif regionEspecial%4 == 3:
            r = r4

        r = r2

        for i in xrange(cant/2):
            x = np.random.uniform(r.minX,r.maxX)
            y = np.random.uniform(r.minY,r.maxY)
            dataset.append(np.array([x,y]))


    if plot:
        sp = plt.subplot(111)
        plt.xlabel("Dataset")
        datazipped = zip(*dataset)

        # regiones en el dataset
        if separados:
            sp.add_patch(patch.Rectangle((10,10),20,20, fill=False, color='r'))
            sp.add_patch(patch.Rectangle((40,10),20,20, fill=False, color='g'))
            sp.add_patch(patch.Rectangle((10,40),20,20, fill=False, color='b'))
            sp.add_patch(patch.Rectangle((40,40),20,20, fill=False, color='y'))
            if mismaDensidad:
                fileName = 'kohonen2-dataset-separado-misma-%d.png' % np.random.randint(1000)
            else:
                fileName = 'kohonen2-dataset-separado-distinta-%d.png' % np.random.randint(1000)
        else:
            sp.add_patch(patch.Rectangle((10,10),20,20, fill=False, color='r'))
            sp.add_patch(patch.Rectangle((30,10),20,20, fill=False, color='g'))
            sp.add_patch(patch.Rectangle((10,30),20,20, fill=False, color='b'))
            sp.add_patch(patch.Rectangle((30,30),20,20, fill=False, color='y'))
            if mismaDensidad:
                fileName = 'kohonen2-dataset-junto-misma-%d.png' % np.random.randint(1000)
            else:
                fileName = 'kohonen2-dataset-junto-distinta-%d.png' % np.random.randint(1000)

        plt.scatter(datazipped[0], datazipped[1], s=10)
        print fileName
        plt.savefig(fileName, bbox_inches='tight')
        plt.show()

    return np.array(dataset)

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

filekey = np.random.randint(1000)
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
    #plt.ion()
    net = NoSupervisedNetwork()

    # generate the data and validation sets
    cant_patterns_training = 400
    #cant_patterns_validation = 400

    data = generateDataset(cant_patterns_training, plot=False, mismaDensidad=True)

    validation = generateDataset(cant_patterns_training, plot=False, mismaDensidad=True)

    net.trainNetwork(data, validation)
    print "Final weights: %s" % net.W

    net.rotularYValidar(data, validation, [r1,r2,r3,r4])

    net.plotWeights()
