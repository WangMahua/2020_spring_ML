#! /usr/bin/env python3
# coding=UTF-8
# This Python file uses the following encoding: utf-8

# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from random import uniform
from scipy.special import erf,erfinv
import math 

MARGIN = 0.04


def s_e(mean,variance,margin):

    now_mean=0
    now_variance=0
    data_number=0
    now_sum=0
    now_sum_2=0
    
    while  abs(now_mean-mean)>margin or abs(now_variance-variance)>margin:  
        i =  trunc_gauss(mean,variance**0.5)
        print("Add data point:"+str(i))
        data_number+=1
        now_sum+=i
        now_mean=now_sum/data_number
        now_sum_2+=((i-now_mean)**2)
        now_variance=now_sum_2/(data_number)
        print("Mean = "+str(now_mean)+" Variance = "+str(now_variance)+"\n") 
    print("end")

def trunc_gauss(mu, sigma,xmin=np.nan,xmax=np.nan):
    if np.isnan(xmin):
        zmin=0
    else:
        zmin = erf((xmin-mu)/sigma)
    if np.isnan(xmax):
        zmax=1
    else:
        zmax = erf((xmax-mu)/sigma)
    y = uniform(zmin,zmax)
    z = (2**0.5)*erfinv(2*y-1)
    return mu + z*sigma

def main():
    mean=input('mean:')
    variance=input('variance:')
    print("Data point source function: N("+str(mean)+", "+str(variance)+")")   
    s_e(mean,variance,MARGIN)
    
if __name__ == '__main__':
    main()
