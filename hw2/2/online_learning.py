#! /usr/bin/env python
from __future__ import print_function
from __future__ import division

import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.stats import beta
from scipy.stats import binom
from scipy.special import gamma


def pdf(a, b, x):
    y = []
    for i in x:
        Z = gamma(a + b ) / gamma(a) / gamma(b)
        beta_f = Z*(i**(a-1))*((1-i)**(b-1))
        y.append(beta_f)
    return y


def likelihood(a,b,x):
    y = []
    for i in x:
        y.append((i**a)*((1-i)**b))
    return y 

def beta_binomial_conjugation(case,prior_a,prior_b):
    posterior_a = 0
    posterior_b = 0
    likelihood = 0
    for letter in case:
        if letter is '1':
            posterior_a+=1
        else:
            posterior_b+=1

    if prior_a!=0 and prior_b!=0:
        p = (posterior_a)/(posterior_a+posterior_b)
    else:
        p = 0.5 
    likelihood = combination(len(case),posterior_a)*(p**(posterior_a))*((1-p)**(posterior_b))
    posterior_a = posterior_a + prior_a
    posterior_b = posterior_b + prior_b 

    return posterior_a,posterior_b,likelihood



def combination(n,r):
    c_num = 1
    for i in range(n,r,-1):
        c_num = c_num * i
    for i in range(1,n-r+1):
        c_num = c_num / i
    return c_num


def print_plot(a,b,a_,b_,c):
    x=[]
    i=0
    while i<1:
        i+=0.01
        x.append(i)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))

    y_1 = pdf(a,b,x)
    y_2 = likelihood(a_-a,b_-b,x)
    y_3 = pdf(a_,b_,x)
    ax1.set_title('prior: a=%d b=%d'%(a,b))
    ax2.set_title('likelihood for case:%s'%str(c))
    ax3.set_title('posterior: a=%d b=%d'%(a_,b_))
    ax1.plot(x, y_1)
    ax2.plot(x, y_2)
    ax3.plot(x, y_3)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig.tight_layout()
    plt.show()
    
    return 0 

def main():
 
    #read data
    f = open("testfile.txt")
    a=input('enter a:')
    b=input('enter b:')
    #build data
    lines = f.readlines()
    data = [] 
    for line in lines:
        curLine=line.strip().split('\n')  
        data.append(curLine[0])

    for case_num in range(len(data)) :
        print('case '+str(case_num+1)+': '+str(data[case_num]))
        next_a,next_b,likelihood=beta_binomial_conjugation(data[case_num],a,b)
        print('Likelihood: '+str(likelihood))
        print('Beta prior:     a = '+str(a)+" b = "+str(b))
        print('Beta posterior: a = '+str(next_a)+" b = "+str(next_b))
        print("\n")
        if a!=0:
            print_plot(a,b,next_a,next_b,data[case_num])
        a = next_a
        b = next_b
    
if __name__ == '__main__':
    main()