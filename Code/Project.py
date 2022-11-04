import numpy
from Tools import *
from Preprocessing import Preprocessing as p
from Classifier import Classifier as c
from Evaluation import Evaluation as e


if __name__ == '__main__':  
        [D_test, L_test] = load("Data/Test.txt")
        D, L = load("Data/Train.txt")
        D,L = shuffle(D,L)
        k = 5
        pi = 0.5
        Cfn = 1
        Cfp = 1
        Score_1 = numpy.array([])
        LTE_total = numpy.array([], dtype=numpy.int32)
        for n in range(0, k):  # K_fold
                DTR, LTR, DTE, LTE = kfold_validation(D, L, n, k)
                #DTR, DTE = p.z_normalization(DTR, DTE)
                #[C, mu] = p.covariance_matrix(DTR)
                #P = p.PCA(C, 7, DTR)
                #DTR = numpy.dot(P.T, DTR)
                #DTE = numpy.dot(P.T, DTE)
                LTE = numpy.array(LTE, dtype=numpy.int32)
                LTE_total = numpy.concatenate((LTE_total, LTE))
                #DTR, DTE = gaussianization(DTR, DTE)
                S1 = c.Binary_Logistic_Regression(DTR, LTR, DTE, 0.1, 1e-5)
                Score_1 = numpy.concatenate((Score_1, S1))
            
    
        print("Binary Logistic regression con pi = 0.5")
        bMin, _ = e.minimum_detection_cost(0.5, Cfn, Cfp, Score_1, LTE_total)
        print(bMin)
        
            