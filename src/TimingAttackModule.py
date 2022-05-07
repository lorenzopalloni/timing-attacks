# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:30:38 2018

@author: Michele
"""
import numpy as np
import random


class TimingAttack():

    keylength=64
    def __init__(self):
        self.keylength=64
        k= [1]+[int(random.random()<=0.5) for i in range(self.keylength-1)]
        k1=[0]+[int(random.random()<=0.5) for i in range(self.keylength-1)]
        self.__secretkey = lambda x : k
        self.__k1 =lambda x: k1

    def setkey(self,k):
        self.__k1 = lambda x: [0]*self.keylength
        self.__secretkey=lambda x: k

    def victimdevice(self,c,mu=1000,sigma=50):
        np.random.seed(c%2**32)
        delay=0
        for i in range(self.keylength):
            delay=delay+np.random.normal(mu,sigma)
            if (self.__secretkey(1)[i])^(self.__k1(1)[i])==1:
                delay=delay+np.random.normal(mu,sigma)
        return delay

    def attackerdevice(self,c,d,mu=1000,sigma=50):
        np.random.seed(c%2**32)
        coupled=True
        delay=0
        for i in range(len(d)):
            delay=delay+np.random.normal(mu,sigma)
            if (d[i]!= ((self.__secretkey(1)[i])^(self.__k1(1)[i]))) & coupled:
                coupled=False
                np.random.seed(random.randint(1,4294967295))
            if d[i]==1:
                delay=delay+np.random.normal(mu,sigma)
        return delay


    def test(self,d, verbose=False):
        #print((self.__secretkey), (self.__k1))
        f= sum([int(((self.__secretkey)(1)[i])^(self.__k1(1)[i])==d[i]) for i in range(len(d))])/len(d)
        if verbose:
            if (f<.75):
                print('Less than 75% of key bits recovered.')
            elif (f<1):
                print('At least 75%, but less than 100% of key bits recovered.')
            else:
                print('100% of key bits recovered.')
        return f
