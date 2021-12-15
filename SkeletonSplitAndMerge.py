#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paolo
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
    
    def __init__(self, mu0, lambda0, alpha0, beta0, alpha):
        self.__mu0 = mu0
        self.__lambda0 = lambda0
        self.__alpha0 = alpha0
        self.__beta0 = beta0
        self.alpha = alpha

    def sample_prior(self, size=1):
        sigmasq = self.__P_0[1].rvs(self.__alpha0, scale=self.__beta0, \
            size=size)
        mu = self.__P_0[0].rvs(loc=np.full(size, self.__mu0), \
            scale=sigmasq/self.__lambda0)
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
            lengthS = np.logical_or((self.__C==self.__C[i]),\
                (self.__C==self.__C[j])).sum()-2  
            self.__S=np.empty(lengthS, dtype=int)  # set S
            index = 0
            for k in range(len(self.__X)):
                if (((self.__C[k]==self.__C[i]) or (self.__C[k]==self.__C[j])) and(k!=i) and (k!=j)):
                    self.__S[index] = k
                    index += 1
                   
        def __Claunch(self,i,j):
            if i==j:
                self.__LabI=np.amax(self.__C)+1 #it's useful to record the new label of i at this point
            else:
                self.__LabI=self.__C[i]

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
                [cl,q]=self.__RestrGS(cl,j,1)
                z=0
                for k in range(len(clSplit)-1):                    
                    if k!=i and k!=j and (k in self.__S):
                        clSplit[k]=self.__cl[z]
                        z=z+1
                    else:
                        if k!=i and k!=j and not(k in self.__S):
                            clSplit[k]=C[k]
                p1=1/q
                p2=math.factorial((len(clSplit==self.__LabI)-1))*math.factorial((len(clSplit==j)-1))/math.factorial((len(self.__S)-1))*alpha
                
                indexes_i = [clSplit==self.__LabI]
                data_i = self.__X[indexes_i] 
                p_i=1
                for z in [clSplit==self.__LabI]:           
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_i)
                    p_i=p_i*prob
                indexes_j = [clSplit==j]
                data_j = self.__X[indexes_j] 
                p_j=1
                for z in [clSplit==j]:           
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_j)
                    p_j=p_j*prob
                indexes_i =[self.__C==self.__LabI]
                data_i = self.__X[indexes_i] 
                P_i=1
                for z in indexes_i:          
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_i)
                    P_i=P_i*prob
                p3=p_i*p_j/P_i
                
                AcRa=p1*p2*p3 #acceptance ratio 
                res=self.__MH(AcRa)
                if res==True:
                    return clSplit
                else:
                    return self.__C
            else:
                clMerge=np.empty(len(self.__C))  
                clMerge[i]=self.__C[j]
                clMerge[j]=self.__C[j]
                for k in range(len(clMerge)-1):                    
                    if k!=i and k!=j and (k in self.__S):
                        clMerge[k]=self.__C[j]
                    else:
                        if k!=i and k!=j and not(k in self.__S):
                            clMerge[k]=C[k]
                
                v=np.array(ndmi=len(cl))
                v=cl
                q=1
                for z in range(len(self.__S)):   #fake gibs sampling to compute q(c/cMerge) 
                   indexes_i = self.__S[v==self.__C[self.__S[z]]]
                   data_i = self.__X[indexes_i] 
                   p_i = self.__hierarchy.conditional_pred_lpdf(
                     self.__X[self.__S[z]], data_i)
                   p_i = len(data_i)*p_i
                   v[z]=self.__C[self.__S[z]]
                   q=q*p_i
                
                p1=q
                p2=math.factorial(len(self.__S)-1)/(math.factorial(len(self.__C==self.__LabI))*math.factorial(len(self.__C==j)))*(1/alpha)
                
                indexes_i =[clMerge==j]
                data_i = self.__X[indexes_i] 
                P_i=1
                for z in indexes_i:          
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_i)
                    P_i=P_i*prob
                indexes_i = [self.__C==self.__LabI]
                data_i = self.__X[indexes_i] 
                p_i=1
                for z in indexes_i:          
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_i)
                    p_i=p_i*prob
                indexes_j = [self.__C==j]
                data_j = self.__X[indexes_j] 
                p_j=1
                for z in indexes_j:          
                    prob = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[z], data_j)
                    p_j=p_j*prob
                    
                p3=p_i*p_j/P_i
                
                AcRa=p1*p2*p3 #acceptance ratio
                               
                res=self.__MH(AcRa)
                if res==True:
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
            pass
            
        def __RestrGS(self, cl,j,use=0): #use!=0 to use the function in SplitOrMerge, in order to report also res_prod
            """Compute a single step of the restricted Gibbs Sampling."""
            res_prod=1 #is the probability to have the current vector of state cl
            RandUnifGenerator=ss.uniform(0,1)
            for z in range(len(self.__S)):
                indexes_i = self.__S[cl==self.__LabI]
                data_i = self.__X[indexes_i] 
                p_i = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[self.__S[z]], data_i)
                p_i = len(data_i)*p_i

                indexes_j = self.__S[cl==self.__C[j]]
                data_j = self.__X[indexes_j] 
                p_j = self.__hierarchy.conditional_pred_lpdf(\
                    self.__X[self.__S[z]], data_j)
                p_j = len(data_j)*p_j

                p = p_i / (p_i+p_j)
                r = RandUnifGenerator.rvs(size=1)
                if p>r:
                  cl[z]=self.__LabI
                  res_prod=res_prod*p_i
                else:
                  cl[z]=self.__C[j]
                  res_prod=res_prod*p_j
            if use==0:        
                return cl
            else:
                return cl,prod_res
          
        def __ProposalSwap(i,j):
            pass  
            
        #public methods
        def __init__(self,X,C, abstractHierarchy):
            #instance attributes
            self.__C=C
            self.__X=X
            self.__hierarchy = abstractHierarchy
                          
        def SplitAndMergeAlgo(self,T,K,M,N=2000):
            matrix=np.empty((N, len(self.__X)))
            for n in range(N):
                for k in range(K):
                    RandIntGenerator=ss.randint(0,len(self.__X)) #generate a random number between 0-len(X)-1
                    r = RandIntGenerator.rvs(size=2)  # i and j indeces
                    self.__ComputeS(r[0],r[1])
                    cl=self.__Claunch(r[0],r[1])
                    for k in range(T):
                        cl = self.__RestrGS(cl,r[1])
                    self.__C=self.__SplitOrMerge(cl,r[0],r[1])
                    
                for m in range(M):
                    self.__C=FullGS()
                
                matrix[n, :] = self.__C.copy()
               


# This snippet of code generates the data and calls the algorithm, to test it.
n_clusters = 10
data_size = 5000
distribution = NNIGHierarchy(0, 0.0001, 100, 1, 1)
parameters = distribution.sample_prior(size=n_clusters)
parameters = np.column_stack(parameters)

parameters_choice = np.random.default_rng().choice(parameters, size=data_size)
data = ss.norm.rvs(loc =parameters_choice[:,0], scale =parameters_choice[:,1])

# These two lines save a plot of the data.
#sns.kdeplot(data)
#plt.savefig("data.png")

labels = np.full(data_size, 1)

labels_samples = SplitAndMerge(data, labels, distribution).\
    SplitAndMergeAlgo(5, 1, 1)