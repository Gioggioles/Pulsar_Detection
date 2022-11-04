import numpy
import scipy
import scipy.optimize
from Tools import *
from Preprocessing import Preprocessing as p


def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):
        # Compute and return the objective function value using DTR,
        w, b = mcol(v[0:M]), v[-1]
        S = numpy.dot(w.T, DTR) + b  #score for my Training Set
        cxe = numpy.logaddexp(0, -S*Z).mean()  #Cross entropy
        return cxe + 0.5 *l* numpy.linalg.norm(w)**2
    return logreg_obj

def feature_expansion(D):
    expansion = []
    for i in range(D.shape[1]):
        v = numpy.reshape(numpy.dot(mcol(D[:, i]), mrow(D[:, i])), (-1,1), order = 'F')
        expansion.append(v)
    return numpy.vstack((numpy.hstack(expansion), D))

def logpdf_GMM(X, gmm):
        S = numpy.zeros((len(gmm), X.shape[1]))
        for g in range(len(gmm)):
            S[g, :] = numpy.log(gmm[g][0]) + logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        logdens = numpy.logsumexp(S, axis=0)
        return logdens


def GMM_LBG(X, alpha, component, psi):
    gmm = [(1, mcol(X.mean(1)), compute_empirical_cov(X))]
    
    while len(gmm) <= component:
        
        
        gmm = GMM_EM(X, gmm, psi)
                
        if len(gmm) == component:
            break
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newGmm.append((w/2, mu + d, sigma))
            newGmm.append((w/2, mu - d, sigma))
        gmm = newGmm
            
    return gmm

def GMM_EM(X,gmm, psi):
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]
        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G,N))
            for g in range(G):
                SJ[g, :] = numpy.log(gmm[g][0]) + logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
            SM = numpy.logsumexp(SJ, axis=0)
            llNew = SM.sum()/N
            P = numpy.exp(SJ- SM)
            gmmNew = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                #Sigma *= numpy.eye(Sigma.shape[0])
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s<psi] = psi
                Sigma = numpy.dot(U, mcol(s)*U.T)
                gmmNew.append((w, mu, Sigma))
            gmm = gmmNew
        return gmm  

def GMM_EM_Tied(X,gmm, psi):  #NON TOCCARE
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]
        D = X.shape[0]
        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G,N))
            for g in range(G):
                SJ[g, :] = numpy.log(gmm[g][0]) + logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
            SM = scipy.special.logsumexp(SJ, axis=0)
            llNew = SM.sum()/N
            P = numpy.exp(SJ- SM)
            gmmNew = []
            tiedSigma = numpy.zeros((D, D))
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                #Sigma *= numpy.eye(Sigma.shape[0])
                tiedSigma += Z * Sigma
                gmmNew.append((w, mu))
            gmm = gmmNew
            tiedSigma = tiedSigma / N
            U, s, _ = numpy.linalg.svd(tiedSigma)
            s[s<psi] = psi
            tiedSigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew = []
            for i in range(G):
                (w, mu) = gmm[i]
                gmmNew.append((w, mu, tiedSigma))
            gmm = gmmNew
            return gmm

class Classifier: 

    def computeGMM(DTR, LTR, DTE, alpha, component, psi = 0.01):

        classi = 2

        log_dens = []
        
        for i in range(classi):
            DTR_i = DTR[:, LTR == i]
            gmm_ci = GMM_LBG(DTR_i, alpha, component, psi)
            logDensity_ci = logpdf_GMM(DTE, gmm_ci)
            log_dens.append(logDensity_ci)



        log_dens = numpy.vstack(log_dens)
        llr = log_dens[1, :] - log_dens[0, :]
        return llr
    
    def Multivariant_Gaussian_Model(DTR, LTR, DTE):
        C0, mu0 = p.covariance_matrix(DTR[:, LTR==0]) 
        C1, mu1 = p.covariance_matrix(DTR[:, LTR==1])
        L0 = logpdf_GAU_ND(DTE, mu0, C0)
        L1 = logpdf_GAU_ND(DTE, mu1, C1)
        llr = L1-L0
        return llr  #array of class posterior probabilities 
    

    def Naive_Bayes_Model(DTR, LTR, DTE):  #diagonal
        C0, mu0 = p.covariance_matrix(DTR[:, LTR==0]) 
        C1, mu1 = p.covariance_matrix(DTR[:, LTR==1])
        
        C0 = numpy.eye(C0.shape[0]) * C0
        C1 = numpy.eye(C1.shape[0]) * C1    
        
        L0 = logpdf_GAU_ND(DTE, mu0, C0)
        L1 = logpdf_GAU_ND(DTE, mu1, C1)

        llr = L1-L0
        return llr
    
    
    def Naive_Bayes_Tied(DTR, LTR, DTE):
        C = 0
        C0, mu0 = p.covariance_matrix(DTR[:, LTR==0]) 
        C1, mu1 = p.covariance_matrix(DTR[:, LTR==1])
        
        C0 = numpy.eye(C0.shape[0]) * C0
        C1 = numpy.eye(C1.shape[0]) * C1
        
        C = C + (C0.shape[1] * C0) 
        C = C + (C1.shape[1] * C1)
        n = C0.shape[1] + C1.shape[1] 
        C = C / n
        
        L0 = numpy.exp(logpdf_GAU_ND(DTE, mu0, C))
        L1 = numpy.exp(logpdf_GAU_ND(DTE, mu1, C)) 
        
        llr = L1-L0
        return llr
        
    
    def Tied_Covariance(DTR, LTR, DTE):
        C = 0
        C0, mu0 = p.covariance_matrix(DTR[:, LTR==0]) 
        C1, mu1 = p.covariance_matrix(DTR[:, LTR==1])
        
        C = C + (C0.shape[1] * C0) 
        C = C + (C1.shape[1] * C1)
        n = C0.shape[1] + C1.shape[1] 
        C = C / n
        
        L0 = numpy.exp(logpdf_GAU_ND(DTE, mu0, C))
        L1 = numpy.exp(logpdf_GAU_ND(DTE, mu1, C)) 
        
        llr = L1-L0
        return llr
        
    
    def Binary_Logistic_Regression(DTR, LTR, DTE, p, l):
        lamb = l
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamb)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        STE = numpy.dot(_w.T, DTE) + _b - numpy.log(p/(1-p))
        
        return STE
        
    
    def Quadratic_Logistic_Regression(DTR, LTR, DTE, p,l):
        DTR = feature_expansion(DTR)
        DTE = feature_expansion(DTE)
        STE = Classifier.Binary_Logistic_Regression(DTR, LTR, DTE, p, l)
        return STE    
        
        
    def train_SVM_Linear(DTR, LTR, DTE, p, C = 1, K = 0):
        DTREXT = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1])) * K])
        Z = LTR * 2 - 1

        G = numpy.dot(DTREXT.T, DTREXT)
        H = mcol(Z)* mrow(Z) *G
        nT = DTR[:, LTR == 0].shape[1] / DTR.shape[1]
        C_t = C * p/nT
        C_f = C * (1-p)/(1-nT)
        
        bound = []
        for i in range(0, DTR.shape[1] ):
            if LTR[i] == 0: 
                bound.append((0,C_t))
            else:
                bound.append((0,C_f))

        def J_D(alpha):
            LD1 = numpy.dot(H, mcol(alpha))
            LD = numpy.dot(mrow(alpha), LD1)
            return -0.5 * LD.ravel() + alpha.sum(), -LD1.ravel() + numpy.ones(
                alpha.size)  # return the function and the gradient

        def L_D(alpha):
            loss, grad = J_D(alpha)
            return -loss, -grad

        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(L_D, numpy.zeros(DTR.shape[1]), bounds=bound)

        wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))
        
        w = wStar[0:-1]
        b = wStar[-1]
        STE = numpy.dot(mrow(w), DTE) + b

        return STE

    def train_SVM_Kernel(DTR, LTR, DTE, p, C = 1, K = 0):
        Z = LTR * 2 - 1        
        #polynomial
        c = 0
        d = 2
        G = numpy.dot(DTR.T, DTR)
        G = (G + c)**d + K
        H = mcol(Z)*mrow(Z)*G
        
        
        nT = DTR[:, LTR == 0].shape[1] / DTR.shape[1]
        '''
        C_t = C * p/nT
        C_f = C * (1-p)/(1-nT)
        
        bound = []
        for i in range(0, DTR.shape[1]):
            if LTR[i] == 0: 
                bound.append((0,C_t))
            else:
                bound.append((0,C_f))
        '''        
               
       

        
        # Radial Basis Function
        gamma = 0.1
        
        C_t = C * p/nT
        C_f = C * (1-p)/(1-nT)
        
        bound_2 = []
        for i in range(0, DTR.shape[1]):
            if LTR[i] == 0: 
                bound_2.append((0,C_t))
            else:
                bound_2.append((0,C_f))
        
                
        H2 = numpy.zeros((DTR.shape[1], DTR.shape[1]))
        exp = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR) 
        kern = numpy.exp(-gamma*exp) + K
        H2 = numpy.dot(Z,Z.T) * kern

        def J_D(alpha):
            #LD1 = numpy.dot(H, mcol(alpha))
            LD2 = numpy.dot(H2, mcol(alpha))
            #L1 = numpy.dot(mrow(alpha), LD1)
            L2 = numpy.dot(mrow(alpha), LD2)
            return (-0.5 * L2.ravel() + alpha.sum(), -LD2.ravel() + numpy.ones(
                alpha.size)) # return the function and the gradient

        def L_D(alpha):
            loss, grad = J_D(alpha)
            return -loss, -grad

        #alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(L_D, numpy.zeros(DTR.shape[1]), bounds=bound)
        alphaStar2, x2, y2 = scipy.optimize.fmin_l_bfgs_b(L_D, numpy.zeros(DTR.shape[1]), bounds=bound_2)

        s2= numpy.zeros(DTE.shape[1])
        #Radial Basis Function
        for i in range(0, DTE.shape[1]):
            for j in range(0, DTR.shape[1]):
                exp = numpy.linalg.norm(DTE[:, i] - DTR[:, j]) ** 2 * gamma
                kern = numpy.exp(-exp) + K
                s2[i] += alphaStar2[j] * Z[j] * kern
                

        '''        
        s = numpy.zeros(DTE.shape[1])
        G = numpy.dot(DTR.T, DTE)
        G = (G + c)**d + K
        for i in range(0, DTE.shape[1]):
            for j in range(0, DTR.shape[1]):
                s[i] += alphaStar[j] * Z[j] * G[j,i]
        '''
        
        #return s, s2
        return s2

        
    