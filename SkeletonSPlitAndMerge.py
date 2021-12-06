#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paolo
"""

import numpy as np
import scipy.stats as ss

class AbstractHierarchy(object):
    """Abstract class that represents the mixture model.

    Notation for the model:
    y_1, ... y_n ~ sum(w_h * k( | theta_h) for h from 1 to M) 
    theta_1, ... theta_M ~ P_0
    """

    def sample_prior(self, size=1):
        """Sample from the prior distribution: generate theta_h ~ P_0."""
        pass

    def compute_posterior_hypers(self, data):
        """Compute and return posterior hypers given data."""
        pass

    def sample_full_conditional(self, data, size=1):
        """Sample from the full conditional distribution: generate theta_h 
        from p(theta_h | ...), which is proportional to 
        P_0(theta_h)* product(k(y_i|theta_h) for all i in cluster h)
        where k is the likelihood. 

        Parameter: 
        data -- a Numpy array containing all the y_i of cluster h.
        """
        pass

    def prior_pred_lpdf(self, x):
        """When k and P_0 are conjugate, computes the marginal/prior 
        predictive distribution in one point:
        m(x) = integral(k(x|theta)P_0(dtheta))
        """
        pass

    def conditional_pred_lpdf(self, x, data):
        """When k and P_0 are conjugate, computes the conditional predictive 
        distribution in one point:
        m(x) = integral(k(x|theta)P_0(dtheta|{y_i: c_i=h}))
        """
        pass


class NNIGHierarchy(AbstractHierarchy):
    """Implement the Normal-Normal-InverseGamma model, i.e. a hierarchical
    model where data are distributed according to a normal likelihood, the
    parameters of which have a Normal-InverseGamma centering distribution.
    That is: 
    k(y_i | mu, sigma) ~ N(mu, sigma^2)
    (mu, sigma^2) ~ P_0 = N-IG(mu0, lambda0, alpha0, beta0)
    """

    __k = ss.norm
    __P_0 = [ss.norm, ss.invgamma]

    def __init__(self, mu0, lambda0, alpha0, beta0):
        self.__mu0 = mu0
        self.__lambda0 = lambda0
        self.__alpha0 = alpha0
        self.__beta0 = beta0

    def sample_prior(self, size=1):
        sigmasq = self.__P_0[1].rvs(self.__alpha0, scale=self.__beta0, \
            size=size)
        mu = self.__P_0[0].rvs(loc=np.full(size, self.__mu0), \
            scale=sigmasq/self.lambda0)
        return [mu, sigmasq]

    def compute_posterior_hypers(self, data):
        n = len(data)
        mu_n = self.__lambda0/(self.__lambda0+n)*self.__mu0 + \
            n/(self.__lambda0+n)*data.mean()
        alpha_n = self.__alpha0 + n/2
        beta_n = self.__beta0 + 0.5*data.var()*n+ \
            0.5*n*self.__lambda0/(n+self.__lambda0)*(-data.mean())**2
        return mu_n, self.__lambda0, alpha_n, beta_n

    def sample_full_conditional(self, data, size=1):
        mu_n, lambda_n, alpha_n, beta_n = self.compute_posterior_hypers(data)

        sigmasq = self.__P_0[1].rvs(alpha_n, scale= beta_n, size=size)
        mu = self.__P_0[0].rvs(loc=np.full(size, mu_n),\
            scale=sigmasq/lambda_n)

        return mu, sigmasq

    def prior_pred_lpdf(self, x):
        return ss.t.pdf(x, 2*self.__alpha0, loc=self.__mu0,\
            scale =self.__beta0*(1+1/self.__lambda0)/self.__alpha0)
            
    def conditional_pred_lpdf(self, x, data):
        mu_n, lambda_n, alpha_n, beta_n = self.compute_posterior_hypers(data)
        return ss.t.pdf(x, 2*alpha_n, loc=mu_n,\
            scale =beta_n*(1+1/lambda_n)/alpha_n)

#Split and Merge Class
class SplitAndMerge(object):
        
        #private methods 
        def __ComputeS(self,i,j):
           self.__S=np.array()  # set S
           for k in range(len(X)):
               if (((self.__C[k]==self.__C[i]) or (self.__C[k]==self.__C[j])) and(k!=i) and (k!=j)):
                   self.__S.add(k)        
                   
        def __Claunch(self,i,j):
            if i==j:
                self.__LabI=np.amax(self.__C)+1 #it's useful to record the new label of i at this point
            else:
                self.__LabI=C[i]

            cl=np.full(len(self.__S), self.__LabI)
            random_assignment = ss.bernoulli.rvs(0.5, size=len(self.__S))
            cl[random_assignment==1] = self.__C[j]
            return cl       
            
        def __SplitOrMerge(self,cl,i,j):
            if i==j:
                clSplit=np.array(ndmi=len(self.__C)) 
                #we keep the reference of the indeces of C
                clSplit[i]=self.__LabI
                clSplit[j]=self.__C[j]
                cl=self.__RestrGS(cl,j)
                z=0
                for k in range(len(clSplit)-1):                    
                    if k!=i and k!=j and (k in self.__S):
                        clSplit[k]=self.__cl[z]
                        z=z+1
                     else:
                         if k!=i and k!=j and !(k in self.__S):
                             clSplit[k]=C[k]
                q= # compute proposal probability (potrebbe servire registrasi le probabilità dell'ultimo Restrictedgibs sampling)
                AcRa= #acceptance ratio 
                res=self.__MH(AcRa)
                if res=True:
                    return clSplit
                else:
                    return self.__C
                
             else:
                clMerge=np.array(ndmi=len(self.__C))  
                clMerge[i]=self.__C[j]
                clMerge[j]=self.__C[j]
                for k in range(len(clMerge)-1):                    
                    if k!=i and k!=j and (k in self.__S):
                        clMerge[k]=self.__C[j]
                     else:
                         if k!=i and k!=j and !(k in self.__S):
                             clMerge[k]=C[k]
                q= # compute proposal probability (potrebbe servire registrasi le probabilità dell'ultimo Restrictedgibs sampling)
                AcRa= #acceptance ratio                  
                res=self.__MH(AcRa)
                if res=True:
                    return clMerge
                else:
                    return self.__C
                
        def __MH(AcRa):   
            RandUnifGenerator=ss.uniform(0,1)
            r = RandUnifGenerator.rvs(size=1)
            if r<=AcRa:
                return True
            else:
                return False
            
        def __FullGS():
                
            
        def __RestrGS(cl,j):
            """Compute a single step of the restricted Gibbs Sampling."""
            RandUnifGenerator=ss.uniform(0,1)
            for z in range(len(S)):
                indexes_i = self.__S[cl==self.__LabI]
                data_i = self.__X[indexes_i] 
                p_i = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[S[z]], data_i)
                p_i = len(data)*p_i

                indexes_j = self.__S[cl==self.__C[j]]
                data_j = self.__X[indexes_j] 
                p_i = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[S[z]], data_j)
                p_j = len(data)*p_j

                p = p_i / (p_i+p_j)
                r = RandUnifGenerator.rvs(size=1)
                if p>r:
                 cl[z]=self.__LabI
                else:
                  cl[z]=self.__C[j]
            return cl      
          
        def __ProposalSwap(i,j):  
            
        #public methods
        def __init__(self,X,C, abstractHierarchy):
            #instance attributes
            self.__C=C
            self.__X=X
            self.__hierarchy = abstractHierarchy
                          
        def SplitAndMergeAlgo(self,T,K,M,N=2000):
            matrix=np.array(ndim=N)
            for n in range(N):
                for k in range(K):
                    RandIntGenerator=ss.randint(0,len(X)) #generate a random number between 0-len(X)-1
                    r = RandIntGenerator.rvs(size=2)  # i and j indeces
                    self.__ComputeS(r[0],r[1])
                    cl=self.__Claunch(r[0],r[1])
                    for k in range(T):
                        cl = self.__RestrGS(cl,r[1])
                    self.__C=self.SplitOrMerge(cl,i,j)
                    
                for m in range(M):
                    self.__C=FullGS()
                
                matrix[n].add(self.__C)
               
