
#! /usr/bin/env python3
# coding=UTF-8
# This Python file uses the following encoding: utf-8


import numpy as np


resp = np.array([(1, 1, 1), (2, 2, 2)], dtype = float)
data = np.array([(1, 1, 1, 1), (0, 0, 1, 1)], dtype = float)
means = np.array([(0.1, 0.2, 0.3, 0.5), (0.2, 0.3, 0.5, 0.5)], dtype = float)
weights = np.array((0.5,0.3))

K=10
D= 2
initMeans = np.full((K, D), 1.0/K)
print(initMeans)

N = len(data)
D = len(data[0])
K = len(means)
Nk = np.sum(resp, axis=0)
mus = np.empty((K,D))

prob = np.zeros((N, K))



initMeans = np.random.rand(10,D)

for i in range(N):
    for k in range(K):
        prob[i,k] = np.prod((means[k]**data[i])*((1-means[k])**(1-data[i])))
    
print(prob)
prob = prob*weights
print(prob)
row_sums = prob.sum(axis=1)[:, np.newaxis]
print(row_sums)
prob = prob/row_sums
print(prob)

for k in range(K):
        print(resp[:,k][:,np.newaxis])
        print(resp[:,k][:,np.newaxis]*data)
    
        mus[k] = np.sum(resp[:,k][:,np.newaxis]*data,axis=0) #sum is over N data points
        print(mus[k])
        try:
            mus[k] = mus[k]/Nk[k]   
        except ZeroDivisionError:
            print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
            break   

    


# for k in range(K):
#         mus[k] = np.sum(resp[:,k][:,np.newaxis]*data,axis=0) #sum is over N data points
#         try:
#             mus[k] = mus[k]/Nk[k]   
#         except ZeroDivisionError:
#             print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
#             break           

