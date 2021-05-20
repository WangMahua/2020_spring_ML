#! /usr/bin/env python3
# coding=UTF-8
# This Python file uses the following encoding: utf-8

from __future__ import division                                                 
from __future__ import print_function                                                                                                                                                                   
                                                                          
import os                                                                       
import sys                                                                      
import struct                                                                                                                                         
import gzip
import math


image_size = 28
K = 10 #10 class(0~9)
N = 60000 #10 class(0~9)
D = 784 #10 class(0~9)

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


def pixel_convert(x):
    count = 0
    while x > 8 :
        count+=1
        x=x-8
    return count

def load_image_data(image_file):
    image = []
    image_for_discrete = []
    magic_number = image_file.read(4)
    magic_number = int.from_bytes(magic_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
    images_number = image_file.read(4)
    images_number = int.from_bytes(images_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
    rows_number = image_file.read(4)
    rows_number = int.from_bytes(rows_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
    columns_number = image_file.read(4)
    columns_number = int.from_bytes(columns_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
    for i in range(images_number):
        temp_image = []
        temp_image_for_discrete = []
        for j in range(rows_number*columns_number):
            data = image_file.read(1)
            data = int.from_bytes(data, byteorder='big')
            data_for_discrete = convert_pixel(data)
            temp_image.append(data)
            temp_image_for_discrete.append(data_for_discrete)
        image.append(temp_image)
        image_for_discrete.append(temp_image_for_discrete)
    #print(image)  
    return image ,image_for_discrete

def load_label_data(label_file):
    label = []
    magic_number = label_file.read(4)
    magic_number = int.from_bytes(magic_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
    items_number = label_file.read(4)
    items_number = int.from_bytes(items_number, byteorder='big') # byteorder='big':輸入左邊bit為高位，右邊為低位
   
    for i in range(items_number):
        data = label_file.read(1)
        data = int.from_bytes(data, byteorder='big')
        label.append(data)
    return label      


def print_test_image(image_list,label_list):
    data=open("output_image.txt",'w+')
    data.write("Imagination of numbers in Bayesian classifier:\n")
    for i in range(len(image_list)):
        data.write(str(label_list[i])+":\n")
        for j in range(28):
            for k in range(28):
                if image_list[i][k+28*j]<8:
                    data.write("0")
                else:
                    data.write("1")                
            data.write("\n")
        data.write("\n")
    data.write("\n")
    return 0

def convert_pixel(x):
    if x > 127:
        return 1
    else:
        return 0

def prior_probability_for_class(labal_list): # probability for 0~9
    prior = [0,0,0,0,0,0,0,0,0,0]
    number = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(labal_list)):
        number[labal_list[i]]+=1
    for i in range(len(prior)):
        prior[i] = number[i] / len(labal_list)
    return prior,number 


def show(image):
    '''
    Function to plot the MNIST data
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=plt.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

def bernoulli(data, means):
    '''To compute the probability of x for each bernouli distribution
    data = N X D matrix
    means = K X D matrix
    prob (result) = N X K matrix 
    '''
    N = len(data)
    K = len(means)
    prob = [[1 for i in range(K)]for j in range(N)]

    for i in range(N):
        for j in range(K):
            for k in range(D):
                print(means[j][k])
                print(data[i][k])
                temp = (means[j][k]**data[i][k])*((1-means[j][k])**(1-data[i][k]))
                prob[i][j] *= temp
    return prob

def respBernoulli(data, weights, means):
    '''To compute responsibilities, or posterior probability p(z/x)
    data = N X D matrix
    weights = K dimensional vector
    means = K X D matrix
    prob or resp (result) = N X K matrix 
    '''
    #step 1
    # calculate the p(x/means)
    prob = bernoulli(data, means)
    
    #step 2
    # calculate the numerator of the resp.s
    for i in range(len(prob)):
        for j in range(K):
            prob[i][j] = prob[i][j]*weights[j]
    
    #step 3
    # calcualte the denominator of the resp.s
    row_sum = [0 for i in range(len(prob))]
    for i in range(N):
        for j in range(K):
            row_sum[i]+=prob[i][j]
        for j in range(K):
            try:
                prob[i][j] = prob[i][j]/row_sum[i] #division 0   
                
            except ZeroDivisionError :
                #print("Division by zero occured in reponsibility calculations!")
                prob[i][j] = 0
    return prob   

    

def bernoulliMStep(data, resp):
    '''Re-estimate the parameters using the current responsibilities
    data = N X D matrix
    resp = N X K matrix
    return revised weights (K vector) and means (K X D matrix)
    '''
    N = len(data)
    D = len(data[0])
    K = len(resp[0])
    NK = [0 for i in range(K)]
    now_weight = [0 for i in range(K)]
    for i in range(K):
        for j in range(N):
            NK[i]+=resp[j][i]
    mus = []
    temp_resp = [[0 for i in range(1)]for j in range(N)]
    for i in range(K):
        for j in range(N):
            temp_resp[j][0] = resp[j][i] 
        mus_k = multiplicationMatrix(transposeMatrix(temp_resp),data)

        for j in range(D):
            mus_k[0][j] = mus_k[0][j]/NK[i] 

        mus.append(mus_k)
        now_weight =  NK[i]/N           
    
    return (now_weight, mus)

def llBernoulli(data, weights, means):
    '''To compute expectation of the loglikelihood of Mixture of Beroullie distributions
    Since computing E(LL) requires computing responsibilities, this function does a double-duty
    to return responsibilities too
    '''
    N = len(data)
    K = len(means)
    
    resp = respBernoulli(data, weights, means)

    
    ll = 0
    for i in range(N):
        sumK = 0
        for k in range(K):
            temp1 = [0 for m in range(D)]
            for j in range(D):
                temp1[j] += ((means[k][j]**data[i][j])*((1-means[k][j])**(1-data[i][j])))
            
            for i in range(D):
                temp1[i] = math.log(temp1[i])

            sum_ = 0
            for j in range(len(temp1)):
                sum_ += temp1[j]

            sumK += resp[i][k]*(math.log(weights[k])*sum_)
        ll += sumK
    return (ll, resp)


def mixOfBernoulliEM(data, init_weights, init_means, maxiters=1000, relgap=1e-4, verbose=False):
    '''EM algo fo Mixture of Bernoulli Distributions'''
    N = len(data)
    D = len(data[0])
    K = len(init_means)
    
    #initalize
    weights = init_weights[:]
    means = init_means[:]
    ll, resp = llBernoulli(data, weights, means)
    ll_old = ll
    
    for i in range(maxiters):
        if verbose and (i % 5 ==0):
            print("iteration {}:".format(i))
            print("   {}:".format(weights))
            print("   {:.6}".format(ll))
            
        #E Step: calculate resps
        #Skip, rolled into log likelihood calc
        #For 0th step, done as part of initialization
            
        #M Step
        weights, means = bernoulliMStep(data, resp)
        
        #convergence check
        ll, resp = llBernoulli(data, weights, means)
        if np.abs(ll-ll_old)<relgap:
            print("Relative gap:{:.8} at iternations {}".format(ll-ll_old, i))
            break
        else:
            ll_old = ll
            
    return (weights, means)

def experiments(data):
    mean = [[1/K for i in range(D)]for j in range(K)] #init mean
    weight = [1/K for i in range(K)]
    return mixOfBernoulliEM(data,weight,mean)

def em_ex(image,label,number):
    D = len(image[0]) #number of random variable 784
    x = [0 for i in range(784)] #random variable
    finWeights, finMeans = experiments(image)
    return 0

#-----#
def mixOfB_EM(data, init_weights, init_means, maxiters=50, relgap=1e-4, verbose=False):
    ll, resp = likelihoodB(data, init_weights, init_means)
    ll_old = ll
    weights = init_weights
    for i in range(maxiters):
        #if verbose and (i % 5 ==0):
        print("iteration {}:".format(i))
        print("   {}:".format(weights))
        print("   {:.6}".format(ll))
            
        #E Step: calculate resps
        #Skip, rolled into log likelihood calc
        #For 0th step, done as part of initialization
            
        #M Step
        print("Mstep")
        weights, means = Mstep(data, resp)

        #convergence check
        print("likelihoodB")
        ll, resp = likelihoodB(data, weights, means)
        if abs(ll-ll_old)<relgap:
            print("Relative gap:{:.8} at iternations {}".format(ll-ll_old, i))
            break
        else:
            ll_old = ll

    return (weights, means)


def likelihoodB(data, weights, means):
    resp = second_respBernoulli(data, weights, means)

    ll = 0
    temp_sum = [[0 for m in range(K)]for n in range(N)]
    for i in range(N):
        for j in range(K):           
            for k in range(D):
                temp = 1
                temp = (means[j][k]**data[i][k])*((1-means[j][k])**(1-data[i][k]))
                try:
                    temp = math.log(temp)
                except:
                    # print("temp is small") 
                    temp = math.log(0.000000000000000000001) #aviod log(0)
                temp_sum[i][j] +=temp
    for i in range(N):
        sumk = 0
        for j in range(K):
            try :
                sumk +=resp[i][j]*(temp_sum[i][j]+math.log(weights[j]))
            except:
                sumk +=resp[i][j]*(temp_sum[i][j]+math.log(0.0000000000000000001))
        ll+=sumk
    return (ll,resp)



def Mstep(data, resp):
    NK = [0 for i in range(K)]
    for i in range(K): 
        for j in range(N):
            NK[i]+=resp[j][i]  #計算每個機率的分母（0~9）-> lambda那行的分子

    mus = [[0 for i in range(D)]for j in range(K)]
    for i in range(K):
        muk_temp = [[0 for m in range(1)]for n in range(N)]
        resp_temp = [[0 for m in range(len(data[0]))]for n in range(len(data))]
        for j in range(N):
            muk_temp[j][0] = resp[j][i] #[:,newaxis]
            for k in range(D):
                resp_temp[j][k] = data[j][k] * muk_temp[j][0]
    
    for i in range(K):    
        for j in range(len(resp_temp[0])):
            for k in range(N):
                mus[i][j]+=resp_temp[k][j]
            try:
                mus[i][j] = mus[i][j]/NK[i] #計算每一個class為正面時的機率
            except:
                mus[i][j] = 0
    temp_sum = 0
    for i in range(K):
        NK[i] = NK[i] / N #weight
    return NK,mus        
    #return (now_weight, mus)



def second_respBernoulli(data,weight,means): #respBernoulli
    #step 1 # calculate the p(x/means)
    prob = first_Bernoulli(data, means)

    #step 2 # calculate the numerator of the resp.s
    for i in range(N):
        for j in range(K):
            prob[i][j] = prob[i][j]*weight[j]
    #step 3 # calcualte the denominator of the resp.s
    row_sum = [0 for i in range(N)] #N
    for i in range(N):
        for j in range(K):
            row_sum[i]+=prob[i][j]
    for i in range(N):
        for j in range(K):
            try:
                prob[i][j]=prob[i][j]/row_sum[i] #division 0     
            except ZeroDivisionError :
                prob[i][j] = 0
                #print("Division by zero occured in reponsibility calculations!")
    return prob


def first_Bernoulli(data,mean):
    prob = [[1 for i in range(K)]for j in range(N)]
    for i in range(N):
        for j in range(K):
            for k in range(D):
                temp = 1
                temp = (mean[j][k]**data[i][k])*((1-mean[j][k])**(1-data[i][k]))
                prob[i][j] *= temp
    return prob


def em(image):

    initial_mean = [[0.1 for i in range(D)]for j in range(K)] #initial
    initial_weight = [0.1 for i in range(K)]

    return mixOfB_EM(image,initial_weight,initial_mean)

def main():

    #read data
    t_i_f = gzip.open("train-images-idx3-ubyte.gz", "rb")
    train_image_list,image_list_em = load_image_data(t_i_f)
    t_l_f = gzip.open("train-labels-idx1-ubyte.gz", "rb")
    train_label_list = load_label_data(t_l_f)
    #prior,number = prior_probability_for_class(train_label_list)
    #em_ex(image_list_em,train_label_list,number)

    em(image_list_em)
 
if __name__ == '__main__':
    main()

