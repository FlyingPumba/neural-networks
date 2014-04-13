import numpy as np

def getTestSetWithNoise(testset, noiseRate):
    cant_patterns = testset.shape[0]
    new_testset = []
    for i in range(0, cant_patterns):
        # alter the pattern
        new_testset.append([alterPattern(noiseRate, testset[i,0]), testset[i,1]])
    return np.asarray(new_testset)

def alterPattern(noiseRate, pattern):
    cant_values = np.size(pattern)
    new_pattern = np.zeros(pattern.shape)

    for i in range(0,cant_values):
        s = np.random.uniform(0,1)

        if(s<noiseRate):
            # alter the value
            if(pattern[i]==0):
                new_pattern[i] = 1
            else:
                new_pattern[i] = 0
        else:
            # copy the same value
            new_pattern[i] = pattern[i]

    return new_pattern

def rangef(min, max, step):
    while min<max:
        yield min
        min = min+step