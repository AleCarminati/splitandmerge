#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paolo
"""

import numpy as np
import scipy.stats as ss

#Split and Merge Class

class SplitAndMerge(object):
        #class attributes
        t=5  #restricted GB steps
        model={
           "F": f
           "G0": g
             }
        
        #private methods 
        def __ComputeS(self,i,j):
           self.__S=np.array()  # set S
           for k in range(len(X)-1):
               if ((self.__C[k]==self.__C[i]) or (self.__C[k]==self.__C[j]) and(k!=i)and(k!=j)):
                   self.__S.add(k)        
                   
        def __Claunch(self,i,j):
            cl=np.array(ndim=len(self.__S)) #non conterrà i e j tanto sono esclusi dal RGsamplig
            if i==j:
                self.__LabI=np.amax (self.__C)+1 #it's useful to record the new label of i at this point
                self.__LabI=NewLabel
            else:
                self.__LabI=C[i]
            
            RandIntGenerator=ss.randint(0,2)
            r = RandIntGenerator.rvs(size=len(self.__S))
            for k in range(len(self.__S) -1)
               if r[k]==0:
                  cl[k]=self.__LabI
               else:
                   cl[k]=self.__C[j]
            cl=self.__RestrGS(cl,j)
                
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
            RandUnifGenerator=ss.uniform(0,1)
            for k in range(t):
                for z in range(len(cl)-1):
                  p=P() # compute conditional probability of Ck=c given cl-k and X
                  r = RandUnifGenerator.rvs(size=1)
                  if p<r:
                     cl[z]=self.__LabI
                  else:
                      cl[z]=C[j]
              return cl      
          
        def __ProposalSwap(i,j):  
            
        #public methods
        def __init__(self,X,C,count):
            #istance attributes
            self.__C=C
            self.__X=X
                          
        def SplitAndMergeAlgo(self,T,K,M,N=2000):
            
            for n in range(N):
                count=0
                
                for k in range(K):
                    RandIntGenerator=ss.randint(0,len(X)) #generate a random number between 0-len(X)-1
                    r = RandIntGenerator.rvs(size=2)  # i and j indeces
                    self.__ComputeS(r[0],r[1])
                    cl=self.__Claunch(r[0],r[1])
                    CNew=self.SplitOrMerge(cl,i,j)
                    
                for m in range(M):
                    
        
            
        
        
        
        
        
        
        
        
        
        
                  
