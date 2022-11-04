import numpy
import math
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def compute_empirical_cov(X):
    mu = mcol(X.mean(1))
    cov = numpy.dot((X - mu), (X - mu).T)/X.shape[1]
    return cov

def mrow(v):
    return v.reshape((1, v.size))

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                #name = line.split(',')[-1].strip()
                DList.append(attrs[0:-1])
                classe = line.split(',')[-1]
                labelsList.append(classe)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
  

def shuffle(D, L, seed = 0):

    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) 
    D = D[:, idx]
    L = L[idx]
    return D,L

def kfold_validation(D, L, to_eval = 0, k = 10):

    DTR = numpy.empty(shape=[D.shape[0], 0])
    LTR = numpy.empty(shape = 0)
    DTE = numpy.empty(shape=[D.shape[0], 0])
    LTE = numpy.empty(shape = 0)

    D_kfold, L_kfold = kfold_split(D, L, k)
    
    for j in range (0, k):
        if(j != to_eval):
            to_add_data = numpy.array(D_kfold[j])
            to_add_label = numpy.array(L_kfold[j])
            DTR = numpy.hstack((DTR, to_add_data))
            LTR = numpy.hstack((LTR, to_add_label))
    
        else :
            to_add_data = numpy.array(D_kfold[j])
            to_add_label = numpy.array(L_kfold[j])
            DTE = numpy.hstack((DTE, to_add_data))
            LTE = numpy.hstack((LTE, to_add_label))
    

    return DTR, LTR, DTE, LTE

def kfold_split(D, L, k = 10):
    D_split = []
    L_split = []
    
    for i in range(0, k):
        if i == k-1:
            D_split.append(D[:, int(i/k*D.shape[1]):])
            L_split.append(L[int(i/k*D.shape[1]):])
        else:
            D_split.append(D[:, int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
            L_split.append(L[int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
    return D_split, L_split
    
def logpdf_GAU_ND(X, m, C):
    Y = []
    inv = numpy.linalg.inv(C)
    _, temp = numpy.linalg.slogdet(C)
    M = X.shape[0]
    c =  M/2 * math.log(2 * math.pi) 
    for i in range(X.shape[1]):
        vett = X[:, i:i+1]
        t = -c - temp/2
        t = t - (1/2 * (numpy.dot(numpy.dot((vett - m).T, inv), (vett - m))))
        Y.append(t)
    return numpy.array(Y).ravel()   

def plot_features(D, L):
    D0 = D[:, L == 0]
    D1 = D[: ,L == 1]

    for i in range(D.shape[0]):
        figure = plt.figure()
        plt.title(f'feature_{i}')
        plt.hist(D0[i, :], bins=100, density=True, alpha=0.6, label = 'negative', color='b', edgecolor = 'black')
        plt.hist(D1[i, :], bins=100, density=True, alpha=0.6, label = 'positive', color='r', edgecolor = 'black')
        plt.legend(loc = 'best')
        plt.show()
        plt.close(figure)

