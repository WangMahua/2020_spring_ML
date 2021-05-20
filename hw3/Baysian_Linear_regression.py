#! /usr/bin/env python3
# coding=UTF-8
# This Python file uses the following encoding: utf-8

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from random import uniform
from scipy.special import erf,erfinv
import math 


now_mean=0
now_variance=0
data_number=0
now_sum=0
now_sum_2=0

def trunc_gauss(mu, sigma):
    y = uniform(0,1)
    z = (2**0.5)*erfinv(2*y-1)
    return mu + z*sigma

def poly_basis_data(n,a,w):
    y = 0
    x = uniform(-1,1)
    e=trunc_gauss(0,a**0.5)
    for i in range(n):
        y +=w[i]*(x**i) 
    return x,y + e

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

def LSE(A,b,lambda_LSE):
    A_t=transposeMatrix(A)
    G=multiplicationMatrix(A_t,A)
    if lambda_LSE != 0:
        for i in range(len(G)):
            G[i][i]+=lambda_LSE

    G_1=getMatrixInverse(G)
    G=multiplicationMatrix(G_1,A_t)
    x=multiplicationMatrix(G,b)

    return x 

def append_new_data(A_,n,x,y):
    new_data = [0 for i in range(n) ]
    for i in range(n):
        new_data[i]=x**i
    A_.append(new_data)    
    return A_
 

def mul_matrix_2(A,n):
    print(n)
    print(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j]*n
    print(A)
    return A  

def plot_fitting_line(x):
    f_c = []
    c = np.linspace(-2,2,num=100)
    for m in range(len(c)):
        sum = 0
        for n in range(len(x)):
            coef = x[n]
            x_d=c[m]**n
            sum += coef*x_d
        f_c.append(sum)
    return c,f_c

def plot_v_line(mean,p_covariance,a):
    f_c = []
    f_c_ = []
    c = np.linspace(-2,2,num=100)
    X=[[0 for row in range(len(mean[0]))] for col in range(len(mean))]
    for m in range(len(c)):
        sum = 0
        for n in range(len(mean)):
            coef = mean[n][0]
            x_d=c[m]**n
            sum += coef*x_d
        for i in range(len(mean)):
            X[i][0]=c[m]**i
        transposeMatrix(X)
        lambda_=multiplicationMatrix(multiplicationMatrix(transposeMatrix(X),getMatrixInverse(p_covariance)),X)
        lambda_ = lambda_[0][0]+a
        lambda_ +=lambda_**0.5
        sum_=sum-lambda_
        sum+=lambda_
        f_c.append(sum)
        f_c_.append(sum_)
    return c,f_c,f_c_

def draw_plot(x,y,w,mean,c,a,m10,m50,c10,c50,c_o):
    fig , ax = plt.subplots()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
 
    #1
    ax1 = plt.subplot(221)
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-20,20])
    ax1.title.set_text('Ground truth')  
    x_,y_=plot_fitting_line(w)
    w_new=[[0 for row in range(1)] for col in range(len(mean))]
    for i in range(len(mean)):
        w_new[i][0]=w[i] 
    p_x,p_y,p_y_ = plot_v_line(w_new,c,a)
    ax1.plot(x_,y_,'k')
    ax1.plot(p_x,p_y,'r')
    ax1.plot(p_x,p_y_,'r')

    #2
    ax2 = plt.subplot(222)
    ax2.set_xlim([-2,2])
    ax2.set_ylim([-20,20])
    ax2.title.set_text('Predict Result')
    for i in range(len(mean)):
        w_new[i]=mean[i][0]   
    x_,y_=plot_fitting_line(w_new)
    p_x,p_y,p_y_ = plot_v_line(mean,c,a)
    ax2.plot(x_,y_,'k')
    ax2.plot(p_x,p_y,'r')
    ax2.plot(p_x,p_y_,'r')
    ax2.scatter(x, y, marker='o',s =5)

    #3
    ax3 = plt.subplot(223)
    ax3.set_xlim([-2,2])
    ax3.set_ylim([-20,20])
    ax3.title.set_text('After 10 incomes')
    for i in range(len(m10)):
        w_new[i]=m10[i][0]
    x_,y_=plot_fitting_line(w_new)
    p_x,p_y,p_y_ = plot_v_line(m10,c10,a)
    ax3.plot(x_,y_,'k')
    ax3.plot(p_x,p_y,'r')
    ax3.plot(p_x,p_y_,'r')
    if len(x)>10:
        x_=x[:10]
        y_=y[:10]
        ax3.scatter(x_, y_, marker='o',s =5) 

    #4
    ax4 = plt.subplot(224)
    ax4.set_xlim([-2,2])
    ax4.set_ylim([-20,20])
    ax4.title.set_text('After 50 incomes')
    for i in range(len(m50)):
        w_new[i]=m50[i][0]
    x_,y_=plot_fitting_line(w_new)
    p_x,p_y,p_y_ = plot_v_line(m50,c50,a)
    ax4.plot(x_,y_,'k')
    ax4.plot(p_x,p_y,'r')
    ax4.plot(p_x,p_y_,'r')
    if len(x)>10:
        x_=x[:50]
        y_=y[:50]
        plt.scatter(x_, y_, marker='o',s =5)
    plt.show()
    return 0 


def mean_and_v(x,y):
    global now_mean
    global now_variance
    global data_number
    global now_sum
    global now_sum_2

    data_number+=1
    now_sum+=x
    now_mean=now_sum/data_number
    now_sum_2+=((x-now_mean)**2)
    now_variance=now_sum_2/(data_number)
    return now_mean,now_variance

def Bayesian(b_,n,a,w):
    A=[]
    y=[]
    draw_x=[]

    mean_prior=[[0 for i in range(1)]for j in range(n)]
    mean_10=[0 for i in range(n)]
    mean_50=[0 for i in range(n)]
    covariance_matrix=[[0 for i in range(n)] for j in range(n)]
    for i in range(len(covariance_matrix)):
        covariance_matrix[i][i]=b_
    c_o=covariance_matrix

    stop_flag=0
    add_new_data_flag=0
    
    while stop_flag==0:
        #input data 
        x,y_=poly_basis_data(n,a,w)
        data = [y_]
        draw_x.append(x)
        y.append(data)
        A=append_new_data(A,n,x,y) 
        a_1=1/a
        #calculate mean and variance
        if add_new_data_flag==0:
            new_covariance_matrix=plus_matrix(mul_matrix_2(multiplicationMatrix(transposeMatrix(A),A),a_1),covariance_matrix)
        else:
            new_covariance_matrix=plus_matrix(mul_matrix_2(multiplicationMatrix(transposeMatrix(A),A),a_1),getMatrixInverse(covariance_matrix))
        
        new_mean = plus_matrix(mul_matrix_2(multiplicationMatrix(transposeMatrix(A),y),a_1),multiplicationMatrix(getMatrixInverse(covariance_matrix),mean_prior))    
        new_mean = multiplicationMatrix(getMatrixInverse(new_covariance_matrix),new_mean)

        #update
        mean_prior=new_mean
        covariance_matrix=new_covariance_matrix

        #print 
        print("Postirior mean:")
        for i in range(len(mean_prior)):
            print(str(mean_prior[i])+"\t")
        print('')
        print("Posterior variance:")
        i_c=getMatrixInverse(covariance_matrix)
        for i in range(len(covariance_matrix)):
            for j in range(len(covariance_matrix[i])):
                print(format(i_c[i][j],'.8f'),end="\t")
            print(' ')
        m,v=mean_and_v(x,y)
        print("Predictive distribution ~ N("+str(m)+","+str(v)+")")

        #check
        stop_flag=0
        print('---')
        for i in range(len(mean_prior)):
            if abs(mean_prior[i][0]-w[i])>0.04:
                stop_flag=0

        #draw
        add_new_data_flag+=1
        if add_new_data_flag==10:
            mean_10=mean_prior
            c_10=covariance_matrix
        if add_new_data_flag==50:
            mean_50=mean_prior
            c_50=covariance_matrix
        if add_new_data_flag%200==0:
            draw_plot(draw_x,y,w,mean_prior,covariance_matrix,a,mean_10,mean_50,c_10,c_50,c_o)
        
    return 0



def main():

    #input
    # b=input('bias number:')
    # n=input('n:')
    # a=input('a:')
    # w=[]
    # for i in range(n):
    #     w.append(input('w['+str(i)+']:'))

    #test
    b=1
    n=3
    a=3
    w=[1,2,3]

    #Bayesian
    Bayesian(b,n,a,w)


    
if __name__ == '__main__':
    main()
