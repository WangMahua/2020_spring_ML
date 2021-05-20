#! /usr/bin/env python3
# coding=UTF-8
# This Python file uses the following encoding: utf-8

from __future__ import division                                                 
from __future__ import print_function                                           
                                                                                                                                                                                                           
import os                                                                       
import sys                                                                      
import math
from numpy import random
from random import uniform
from scipy.special import erf,erfinv
import math 
import matplotlib.pyplot as plt

W_DEGREE = 3

def trunc_gauss(mu, sigma):
    y = uniform(0,1)
    z = (2**0.5)*erfinv(2*y-1)
    return mu + z*sigma

def transposeMatrix(o_matrix):
    t_matrix=[[0 for row in range(len(o_matrix))] for col in range(len(o_matrix[0]))]
    for i in range(len(o_matrix)):
        for j in range(len(o_matrix[0])):
            t_matrix[j][i]=o_matrix[i][j]
    return t_matrix

def multiplicationMatrix(matrix1,matrix2):
    m_matrix=[[0 for row in range(len(matrix2[0]))] for col in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                m_matrix[i][j] += matrix1[i][k]*matrix2[k][j]
    return m_matrix

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def determinantMatrix(o_matrix):
    if len(o_matrix) == 2:
        return o_matrix[0][0]*o_matrix[1][1]-o_matrix[0][1]*o_matrix[1][0]
    determinant=0
    for c in range(len(o_matrix)):
        determinant += ((-1)**c)*o_matrix[0][c]*determinantMatrix(getMatrixMinor(o_matrix,0,c))
    return determinant
        
def getMatrixInverse(m):
    determinant = determinantMatrix(m)
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * determinantMatrix(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def plus_matrix(A,B):
    C = [[0 for row in range(len(A[0]))] for col in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
    return C 

def minus_matrix(A,B):
    C = [[0 for row in range(len(A[0]))] for col in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] - B[i][j]
    return C 

def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i][0]**2
    total = total**0.5
    return total 

def generate_sample_data(N,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2):
    D1=[]
    D2=[]
    for i in range(N):
        data1=[]
        data2=[]
        data1.append(trunc_gauss(mx1,vx1**0.5))
        data1.append(trunc_gauss(my1,vy1**0.5)) #y
        data1.append(1) #label original data
        data2.append(trunc_gauss(mx2,vx2**0.5))
        data2.append(trunc_gauss(my2,vy2**0.5)) #y
        data2.append(2) #label original data
        D1.append(data1)
        D2.append(data2)

    design_matrix = []
    y = []
    for i in range(2*N):
        data=[]
        data_y=[]
        if i < N:
            data_y.append(D1[i][1])
            for j in range(W_DEGREE):
                data.append(D1[i][0]**j)
        else:
            data_y.append(D2[i-50][1])
            for j in range(W_DEGREE):
                data.append(D2[i-50][0]**j)           
        design_matrix.append(data)
        y.append(data_y)
    return D1,D2,design_matrix,y

def gradient_ascent(design_matrix,y,N):
    g = [[1 for i in range(1)]for j in range(W_DEGREE)]
    w = [[0.1 for i in range(1)]for j in range(W_DEGREE)]
    m = 0
    while norm(g)>0.2:
        m+=1
        h = multiplicationMatrix(design_matrix,w)
        e = [[0 for i in range(1)]for j in range(2*N)]
        for i in range(2*N):
            e[i][0] = y[i][0] - 1/(1+math.exp(-h[i][0]))
        g = multiplicationMatrix(transposeMatrix(design_matrix),e)
        w_new = plus_matrix(w,g)
        w = w_new
        print(h)
        print(e)
        print(g)
        print(w)
        print("---")
    return w_new

def Hessian_matrix(w,design_matrix):
    H=[[0 for i in range(len(design_matrix))]for j in range(len(design_matrix))]
    for i in range(len(design_matrix)):
        h = multiplicationMatrix(design_matrix,w)
        H[i][i]=math.exp(-h)/(1+math.exp(-h))**2
    return H


def draw_plot(D1,D2):
    D1_x_plot = []
    D1_y_plot = []
    D2_x_plot = []
    D2_y_plot = []
    for i in range(len(D1)):
        D1_x_plot.append(D1[i][0])
        D1_y_plot.append(D1[i][1])
    for i in range(len(D2)):
        D2_x_plot.append(D2[i][0])
        D2_y_plot.append(D2[i][1])
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
    ax1.scatter(D1_x_plot, D1_y_plot, marker='o', c="red",s =5)
    ax1.scatter(D2_x_plot, D2_y_plot, marker='o', c="blue",s =5)    
    plt.show()
    return 0

def newton_method(design_matrix,y,N):
    g = [[1 for i in range(1)]for j in range(W_DEGREE)]
    w_new = [[1.5 for i in range(1)]for j in range(W_DEGREE)]
    while norm(g)>0.2:
        w = w_new
        h = multiplicationMatrix(design_matrix,w)
        e = [[1 for i in range(1)]for j in range(2*N)]
        for i in range(2*N):
            e[i][0] = y[i][0] - 1/(1+math.exp(-h[i][0]))
        print(e)
        print(h)
        g = multiplicationMatrix(transposeMatrix(design_matrix),e)
        w_new = plus_matrix(w,g)
        print(w_new)
       
    return w_new

def main():

    # read data
    # N=input('number of data points:')
    # mx1=input('mx1:')
    # my1=input('vx1:')
    # mx2=input('my1:')
    # my2=input('vy1:')
    # vx1=input('vx1:')
    # vy1=input('vx2:')
    # vx2=input('my2:')
    # vy2=input('vy2:')

    N=50
    mx1=1
    my1=1
    mx2=10
    my2=10
    vx1=2
    vy1=2
    vx2=2
    vy2=2
    D1,D2,design_matrix,y=generate_sample_data(N,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2)
    print(D1)
    print(D2)
    print(design_matrix)
    print(y)
    #draw_plot(D1,D2)
    w_final = gradient_ascent(design_matrix,y,N)
    


if __name__ == '__main__':
    main()
