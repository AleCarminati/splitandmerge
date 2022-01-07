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

seed = 265815667228932327047115325544902682949
# Command to set the random seed to a casual value
seed = np.random.SeedSequence().entropy

# Command to set the random seed to a fixed value for reproducibility of the
# experiments.
rng = np.random.default_rng(seed)

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
            size=size, random_state=rng)
        mu = self.__P_0[0].rvs(loc=np.full(size, self.__mu0), \
            scale=sigmasq/self.__lambda0, size=size, random_state=rng)
        print(f"{mu}")
        print("......")
        print(f"{sigmasq}")
        return np.column_stack((mu, sigmasq))

    def compute_posterior_hypers(self, data):
        n = len(data)
        if(n==0):
            raise Exception("Empty data array passed to "+\
                "compute_posterior_hypers")
        mu_n = self.__lambda0/(self.__lambda0+n)*self.__mu0 + \
            n/(self.__lambda0+n)*data.mean()
        alpha_n = self.__alpha0 + n/2
        beta_n = self.__beta0 + 0.5*data.var()*n+0.5*n*self.__lambda0\
            /(n+self.__lambda0)*(self.__mu0-data.mean())**2
        return mu_n, self.__lambda0+n, alpha_n, beta_n

    def sample_full_conditional(self, data, size=1):
        mu_n, lambda_n, alpha_n, beta_n = self.compute_posterior_hypers(data)

        sigmasq = self.__P_0[1].rvs(alpha_n, scale= beta_n, size=size,\
            random_state=rng)
        mu = self.__P_0[0].rvs(loc=np.full(size, mu_n),\
            scale=sigmasq/lambda_n, random_state=rng)

        return np.column_stack((mu, sigmasq))

    def prior_pred_lpdf(self, x):
        """ Note: the formula has been taken from the function marg_lpdf of 
        the file nnig_hierarchy.cc in bayesmix.
        """
        return ss.t.logpdf(x, 2*self.__alpha0, loc=self.__mu0,\
            scale =math.sqrt(self.__beta0*(1+1/self.__lambda0)/self.__alpha0))
            
    def conditional_pred_lpdf(self, x, data):
        if(len(data)==0):
            raise Exception("Empty data array passed to"+\
                "conditional_pred_lpdf.")
        mu_n, lambda_n, alpha_n, beta_n = self.compute_posterior_hypers(data)
        return ss.t.logpdf(x, 2*alpha_n, loc=mu_n,\
            scale =math.sqrt(beta_n*(1+1/lambda_n)/alpha_n))

#Split and Merge Class
class SplitAndMerge(object):
        
        #private methods 
        def __ComputeS(self,i,j):
            lengthS = np.logical_or((self.__C==self.__C[i]),\
                (self.__C==self.__C[j])).sum()-2    
            self.__S=np.empty(lengthS, dtype=int)  # set S
            index = 0
            for k in range(len(self.__C)):
                if ((self.__C[k]==self.__C[i]) or(self.__C[k]==self.__C[j]))\
                    and(k!=i) and (k!=j):
                    if index >= lengthS:
                        raise Exception(f"Index out of bounds. index={index},\
                        lengthS={lengthS} self.__C={self.__C}, i={i}, j={j}")
                    self.__S[index] = k
                    index += 1
                   
        def __Claunch(self,i,j):
            if self.__C[i]==self.__C[j]:
                self.__LabI=np.amax(self.__C)+1 #it's useful to record the new label of i at this point
            else:
                self.__LabI=self.__C[i]

            cl=np.full(len(self.__S), self.__LabI)
            random_assignment = ss.bernoulli.rvs(0.5, size=len(self.__S),\
                random_state = rng)
            cl[random_assignment==1] = self.__C[j]
            return cl       
            
        def __SplitOrMerge(self,cl,i,j):
            if self.__C[i]==self.__C[j]:
                clSplit=np.empty(shape=len(self.__C)) 
                #we keep the reference of the indeces of C
                clSplit[i]=self.__LabI
                clSplit[j]=self.__C[j]
                [cl,q]=self.__RestrGS(cl,i,j,1)
                z=0
                for k in range(len(clSplit)):                    
                    if k in self.__S:
                        clSplit[k]=cl[z]
                        z=z+1
                    else:
                        if k!=i and k!=j and not(k in self.__S):
                            clSplit[k]=self.__C[k]
                p1=1/q
                p2=math.factorial(((clSplit==self.__LabI).sum()-1))*\
                    math.factorial(((clSplit==self.__C[j]).sum()-1))/\
                    math.factorial((len(self.__S)+2-1))*self.__hierarchy.alpha
                
                indexes_i = clSplit==self.__LabI
                data_i = self.__X[indexes_i] 
                p_i=0
                for k,z in enumerate(data_i):    
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_i[0:k])
                    p_i+=prob
                indexes_j = clSplit==self.__C[j]
                data_j = self.__X[indexes_j] 
                p_j=0

                for k,z in enumerate(data_j):   
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_j[0:k])
                    p_j+=prob
                indexes_i = self.__C==self.__C[j]
                data_i = self.__X[indexes_i] 
                P_i=0
                for k,z in enumerate(data_i):   
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_i[0:k])
                    P_i+=prob
                p3=math.exp(p_i+p_j-P_i)
                
                AcRa=min(1,p1*p2*p3) #acceptance ratio 
                res=self.__MH(AcRa)
                if res==True:
                    return clSplit
                else:
                    return self.__C
            else:
                clMerge=np.empty(len(self.__C))  
                clMerge[i]=self.__C[j]
                clMerge[j]=self.__C[j]
                for k in range(len(clMerge)):                    
                    if k in self.__S:
                        clMerge[k]=self.__C[j]
                    else:
                        if k!=i and k!=j and not(k in self.__S):
                            clMerge[k]=self.__C[k]
                
                v=np.empty(shape=len(cl))
                v=cl.copy()
                q=1
                for z in range(len(self.__S)):   #fake gibbs sampling to compute q(c/cMerge) 
                     p_i=self.__ComputeRestrGSProbabilities(v, i, j, z, cluster="i")
                     p_j=self.__ComputeRestrGSProbabilities(v, i, j, z, cluster="j")
                     p=p_i/(p_i+p_j)                    
                     v[z]=self.__C[self.__S[z]]
                     if (v[z]==self.__C[i]):
                         q=q*p
                     else:
                         q=q*(1-p)                
                p1=q
                p2=math.factorial(len(self.__S)+2-1)/\
                    (math.factorial((self.__C==self.__LabI).sum())*\
                    math.factorial((self.__C==self.__C[j]).sum()))*\
                    (1/self.__hierarchy.alpha)
                
                indexes_i = clMerge==self.__C[j]
                data_i = self.__X[indexes_i] 
                P_i=0              
                for k,z in enumerate(data_i):
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_i[0:k])
                    P_i+=prob
                indexes_i = self.__C==self.__LabI
                        
                data_i = self.__X[indexes_i] 
                p_i=0
                for k,z in enumerate(data_i): 
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_i[0:k])
                    p_i+=prob
                indexes_j = self.__C==self.__C[j]
                data_j = self.__X[indexes_j] 
                p_j=0
                for k,z in enumerate(data_j):          
                    if(k==0): #first iteration
                        prob=self.__hierarchy.prior_pred_lpdf(z)
                    else:   
                        prob = self.__hierarchy.conditional_pred_lpdf(z, data_j[0:k])
                    p_j+=prob
                    
                p3=math.exp(-p_i-p_j+P_i)
                
                AcRa=min(1,p1*p2*p3)#acceptance ratio
                               
                res=self.__MH(AcRa)
                if res==True:
                    return clMerge
                else:
                    return self.__C
                
        def __MH(self, AcRa):   
            RandUnifGenerator=ss.uniform(0,1)
            r = RandUnifGenerator.rvs(size=1, random_state=rng)
            if r<=AcRa:
                return True
            else:
                return False
            
        def __FullGS(self):
            '''Compute a single step of full Gibbs sampling on the cluster
            labels.
            '''
            for i in range(len(self.__C)):
                X_without_i = np.delete(self.__X,i)
                C_without_i = np.delete(self.__C,i)
                unique_labels = np.unique(C_without_i)
                unique_labels = np.append(unique_labels, max(unique_labels)+1)
                labels_prob = np.empty(len(unique_labels))
                for j in range(len(unique_labels)):
                    label = unique_labels[j]
                    # Note: np.delete deletes from a copy of the array, not 
                    # from the original array. 
                    data_cluster = X_without_i[C_without_i==label]
                    n_cluster = len(data_cluster)
                    if n_cluster != 0:
                        labels_prob[j] = n_cluster * math.exp(\
                            self.__hierarchy.conditional_pred_lpdf(\
                                self.__X[i], data_cluster))
                    else:
                        labels_prob[j] = math.exp(\
                            self.__hierarchy.prior_pred_lpdf(self.__X[i]))
                
                labels_prob = labels_prob / labels_prob.sum()
                self.__C[i] = rng.choice(unique_labels, \
                    p = labels_prob)

        def __ComputeRestrGSProbabilities(self, cl, i, j, \
            z, cluster="i"):
            """ Auxiliary function for __RestrGS.
            Compute the UNNORMALIZED probabilities for a certain data point
            to be in one of the two clusters considered in the restricted 
            Gibbs sampling. 

            Input:
            cl -- A 1-D Numpy array which contains the cluster labels for the 
                data points in S.
            i -- The index (w.r.t. the entire dataset) of the first point 
                selected for the Split&Merge step. 
            j -- The index (w.r.t. the entire dataset) of the second point 
                selected for the Split&Merge step.
            z -- The index (w.r.t. set S) of the point for which the function 
                is computing the probabilities.
            cluster -- Takes only values "i" and "j". Label that marks which 
                probability to return: the probability for the point indexed
                by z to belong to the cluster proposed for i or to the
                cluster proposed for j.

            Output:
            p -- The probability for the point indexed by z to belong to the
                cluster proposed for i or to the cluster proposed for j 
                (based on the value of input parameter 'cluster'). 
            """
            if cluster!='i' and cluster!='j':
                raise Exception("Unexpected value for the parameter "+\
                    "'cluster' of function __ComputeRestrGSProbabilities.")

            label = 0
            if cluster=='i':
                label = self.__LabI
            else:
                label = self.__C[j]

            indexes = self.__S[cl==label]
            # Remove the considered point from the points used to compute
            # the posterior. 
            indexes = indexes[indexes!=self.__S[z]]
            # Add to the data to consider for the posterior also i or j.
            # This step must be done because cl does not contain i and j.
            if cluster=='i':
                indexes = np.append(indexes, i)
            else:
                indexes = np.append(indexes, j)
            data = self.__X[indexes]

            if(len(data)==0):
                raise Exception("No data points in one of the two "+\
                    "clusters considered for restricted Gibbs sampling. "+\
                    "This is impossible, indeed there should always be at "+\
                    "least i or j in the datapoints.")  
            return len(data)*math.exp(self.__hierarchy.conditional_pred_lpdf(\
                self.__X[self.__S[z]], data))
                
        def __RestrGS(self, cl,i,j,use=0): #use!=0 to use the function in SplitOrMerge, in order to report also res_prod
            """Compute a single step of the restricted Gibbs Sampling."""
            res_prod=1 #is the probability to have the current vector of state cl
            RandUnifGenerator=ss.uniform(0,1)
            for z in range(len(self.__S)):
                p_i = self.__ComputeRestrGSProbabilities(cl, i, j, z, "i")
                p_j = self.__ComputeRestrGSProbabilities(cl, i, j, z, "j")

                p = p_i / (p_i+p_j)
                r = RandUnifGenerator.rvs(size=1, random_state = rng)
                if p>r:
                  cl[z]=self.__LabI
                  res_prod=res_prod*p
                else:
                  cl[z]=self.__C[j]
                  res_prod=res_prod*(1-p)
            if use==0:        
                return cl
            else:
                return cl, res_prod
          
        def __ProposalSwap(self, i,j):
            pass  
            
        #public methods
        def __init__(self,X,C, abstractHierarchy):
            #instance attributes
            self.__C=C
            self.__X=X
            self.__hierarchy = abstractHierarchy
                          
        def SplitAndMergeAlgo(self,T,K,M,N=2000):
            print("Starting Split and Merge Algorithm!")
            print(f"Random seed: {seed}")
            print(f"Iteration: 0/{N}", end="\r")
            matrix=np.empty((N, len(self.__X)), dtype=int)
            for n in range(N):
                for k in range(K):
                    RandIntGenerator=ss.randint(0,len(self.__X)) #generate a random number between 0-len(X)-1
                    r = RandIntGenerator.rvs(size=2, random_state = rng)  # i and j indeces
                    while r[0]==r[1]:
                        r = RandIntGenerator.rvs(size=2)
                    self.__ComputeS(r[0],r[1])
                    cl=self.__Claunch(r[0],r[1])
                    for k in range(T):
                        cl = self.__RestrGS(cl,r[0],r[1])
                    self.__C=self.__SplitOrMerge(cl,r[0],r[1])
                    
                for m in range(M):
                    self.__FullGS()
                
                matrix[n, :] = self.__C.copy()

                print(f"Iteration: {n+1}/{N}", end="\r")

            print("\nCompleted!")
            return matrix
               

def cluster_estimate(chain_result):
    """Compute the cluster estimate starting from the result of a chain.

    Input:
    chain_result -- A bidimensional Numpy array. Each row represents an 
                    iteration of the chain.

    Output:
    estimate -- A Numpy array that contains the final cluster estimate.

    Notes: the code has been copied and adapted from the homonymous function 
    in library bayesmix.  
    """
    n_iter = chain_result.shape[0]
    n_data = chain_result.shape[1]
    mean_diss = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(i-1):
            mean_diss[i,j] = (chain_result[:,i]==chain_result[:,j]).sum()
            mean_diss[j,i] = mean_diss[i,j]
    mean_diss = mean_diss/n_iter

    errors = np.empty(n_iter)
    for k in range(n_iter):
        x = np.empty((n_data, n_data))
        for i in range(n_data):
            x[i,:] = chain_result[k,:]==chain_result[k,i]
        errors[k] = ((x-mean_diss)**2).sum()/2

    return (chain_result[errors==errors.min(),:])[0, :]

if __name__ == "__main__":
    # This snippet of code generates the data and calls the algorithm.
    n_iterations = 100
    n_clusters = 5
    data_size = 100
    prior_distribution = NNIGHierarchy(0, 0.01, 100, 100, 1)
    data_distribution = NNIGHierarchy(0, 0.01, 100, 100, 1)
    parameters = data_distribution.sample_prior(size=n_clusters)

    parameters_choice = rng.integers(0, high=(n_clusters-1),\
        size=data_size)
    #parameters_choice = rng.choice(parameters,\
    #    size=data_size)
    #data = ss.norm.rvs(loc =parameters_choice[:,0],\
    #    scale =parameters_choice[:,1])

    data = ss.norm.rvs(loc =parameters[parameters_choice,0],\
        scale = parameters[parameters_choice,1], random_state = rng)

    # These two lines save a plot of the data.
    #sns.kdeplot(x=data, hue=parameters_choice)
    sns.scatterplot(x=data, y=np.repeat("Split&Merge", len(data)),\
        hue=parameters_choice,palette = "tab10", linewidth=0, legend=None)
    plt.title("Generated data")
    plt.yticks(rotation=90, verticalalignment="center")
    plt.savefig("data.png", dpi=500)
    plt.close()

    labels = np.full(data_size, 1, dtype=int)

    labels_samples = SplitAndMerge(data, labels, prior_distribution).\
        SplitAndMergeAlgo(5, 1, 1, N=n_iterations)

    n_clusters_samples = np.apply_along_axis(lambda x: len(np.unique(x)), 1,\
        labels_samples)
    p = sns.lineplot(x=np.full(n_iterations, 1, dtype=int).cumsum(),\
        y=n_clusters_samples)
    plt.xlabel("Iteration")
    plt.ylabel("Number of clusters")
    plt.savefig("n_clusters.png", dpi=500)
    plt.close()

    clust_estimate = cluster_estimate(labels_samples)
    clust_estimate = ss.rankdata(clust_estimate, method='dense')

    # These two lines save a plot of the data, clustered using Split&Merge.
    #sns.kdeplot(data, hue=clust_estimate)
    sns.scatterplot(x=data, y=np.repeat("Split&Merge", len(data)),\
        hue=clust_estimate,palette = "tab10", linewidth=0, legend=None)
    plt.yticks(rotation=90, verticalalignment="center")
    plt.savefig("data_clustered.png", dpi=500)
    plt.close()
