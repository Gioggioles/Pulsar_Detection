import numpy
import scipy
import pylab
import matplotlib.pyplot as plt
from Tools import *
from Preprocessing import Preprocessing as p
from Classifier import Classifier as c



class Evaluation:
    def optimal_Bayes_decision(pi, Cfn, Cfp, LTE, llr):
        error = 0
        t = -numpy.log(pi*Cfn) + numpy.log((1-pi)*Cfp)
        pred = numpy.int32(llr > t) *1    #casti a 1 o 0 se è maggiore o minore della soglia
        conf = numpy.zeros((2,2))

        for i,j in zip(pred, LTE):
            conf[i,j] += 1        
        
        error += (LTE == pred).sum()
        error = error/LTE.size *100
        return conf, error
    
    
    
    def det_curve(LTE, llr):
        thresholds = numpy.array(llr)
        thresholds.sort()
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
        FNR = numpy.zeros(thresholds.size)
        for idx,t in enumerate(thresholds):
            pred = numpy.int32(llr > t) *1
            conf = numpy.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    conf[i,j] = ((pred == i) * (LTE == j)).sum()
            TPR[idx] = conf[1,1] / (conf[1,1] + conf[0,1])
            FNR[idx] = 1 - TPR[idx]
            FPR[idx] = conf[1,0] / (conf[1,0] + conf[0,0])

        pylab.plot(FPR,FNR)
        pylab.xlabel("FPR")
        pylab.ylabel("FNR")
        pylab.show()
        
        
    def binarytask_evaluation(pi, Cfn, Cfp, llr, LTE, t=None):
            if t == None:
                t = -numpy.log(pi * Cfn) + numpy.log((1 - pi) * Cfp)
            
            pred = numpy.int32(llr > t)  # casti a 1 o 0 se è maggiore o minore della soglia
            conf = numpy.zeros((2, 2))
            
            for i, j in zip(pred, LTE):
                conf[i, j] += 1
            FNR = conf[0,1] / (conf[1,1] + conf[0,1])
            FPR = conf[1,0] / (conf[1,0] + conf[0,0])

            b = pi * Cfn * FNR + (1 - pi) * Cfp * FPR   

            return b


    def normalize_DCF (pi, Cfn, Cfp, llr, LTE, t=None):
            Bdummy = min(pi * Cfn, (1-pi)*Cfp)
            b = Evaluation.binarytask_evaluation(pi, Cfn, Cfp, llr, LTE, t)
            bNormalize = b/Bdummy            
            return bNormalize
         
            

    def minimum_detection_cost(pi, Cfn, Cfp, llr, LTE):
            thresholds = numpy.array(llr)
            thresholds.sort()
            thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
            bMin = numpy.inf

            for t in thresholds:
                pred = numpy.int32(llr > t)
                conf = numpy.zeros((2,2))
                for i in range(2):
                    for j in range(2):
                        conf[i,j] = ((pred == i) * (LTE == j)).sum()
                FNR = conf[0, 1] / (conf[1, 1] + conf[0, 1])
                FPR = conf[1, 0] / (conf[1, 0] + conf[0, 0])

                b = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
                Bdummy = min(pi * Cfn, (1 - pi) * Cfp)
                bNormalize = b / Bdummy
                if(bNormalize < bMin):
                    bMin = bNormalize
                    tMin = t
            return bMin, tMin
                
        
        
    def bayes_error_plots(llr, llr2, LTE):
        effPriorLogOdds = numpy.linspace(-4, 4, 20)
        pi_tilde = 1 / (1 + numpy.exp(-effPriorLogOdds))
        dcf = []
        dcf_min = []
        for pp in pi_tilde:
            dcf.append(Evaluation.normalize_DCF(pp, 1, 1, llr, LTE))
            dcf_min.append(Evaluation.minimum_detection_cost(pp, 1, 1, llr, LTE))
            
        dcf2 = []
        dcf_min_2 = []
        for pp in pi_tilde:
            dcf2.append(Evaluation.normalize_DCF(pp, 1, 1, llr2, LTE))
            dcf_min_2.append(Evaluation.minimum_detection_cost(pp, 1, 1, llr2, LTE))           

        plt.plot(effPriorLogOdds, dcf, label="Log Reg actDCF", color="r")   #Normalize_DCF
        plt.plot(effPriorLogOdds, dcf2, label="SVM Poly actDCF", color="b")   #Normalize_DCF
        plt.plot(effPriorLogOdds, dcf_min, label="LogReg min DCF", color="r", linestyle='--')  #Minimum_Detection_Cost
        plt.plot(effPriorLogOdds, dcf_min_2, label="SVM Poly min DCF", color="b", linestyle='--')  #Minimum_Detection_Cost
        plt.xlabel("log p/(1-p)")
        plt.legend()
        plt.ylabel("DCF")
        plt.ylim([0, 1.2])
        plt.xlim([-4, 4])
        plt.show()
    
    def logreg_obj_wrap_2(DTR, LTR, l):
        Z = LTR * 2.0 - 1.0
        def logreg_obj_2(v):
            # Compute and return the objective function value using DTR,
            w, b = v[0:1], v[-1]
            S = w * DTR + b  #score for my Training Set
            cxe = numpy.logaddexp(0, -S*Z).mean()  #Cross entropy
            return cxe + 0.5 *l* numpy.linalg.norm(w)**2
        return logreg_obj_2


    def Score_calibration(DTR, LTR, DTE, p, l):
        lamb = l
        logreg_obj = Evaluation.logreg_obj_wrap_2(DTR, LTR, lamb)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(2), approx_grad=True)
        _w = _v[0]
        _b = _v[-1]
        STE = _w * DTE + _b - numpy.log(p/(1-p))
        
        return STE