import numpy
import scipy.linalg
import scipy.stats
from Tools import *


class Preprocessing:
    
    def covariance_matrix(D):
        mu = mcol(D.mean(1))
        Dc = D - mu
        C = numpy.dot(Dc, Dc.T)
        C = C / float(D.shape[1])
        return C, mu

    def z_normalization(DTR, DTE):
        DTR_m = numpy.mean(DTR)
        DTR_std = numpy.std(DTR)
        DTR = (DTR-DTR_m)/DTR_std
        DTE = (DTE-DTR_m)/DTR_std
        return DTR, DTE

    def PCA(C, m, D):
        D  = D - D.mean()
        s, U = numpy.linalg.eigh(C)    
        P = U[:, ::-1][:, 0:m]
        return P 
    
    
    def gaussianization(DTR, DTE = None):

        rankDTR = numpy.zeros(DTR.shape)
        for i in range(DTR.shape[0]):
            for j in range(DTR.shape[1]):
                rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
        
        DTR_new = scipy.stats.norm.ppf((rankDTR + 1)/(DTR.shape[1] + 2))

        if DTE is not None :
            rankDTE = numpy.zeros(DTE.shape)
            for i in range(DTE.shape[0]):  
                for j in range(DTE.shape[1]):
                    rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum()

            DTE_new = scipy.stats.norm.ppf((rankDTE + 1)/(DTR.shape[1] + 2)) 

            return DTR_new, DTE_new
            
        return DTR_new 
    
    