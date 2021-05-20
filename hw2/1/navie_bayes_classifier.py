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
            data_for_discrete = pixel_convert(data)
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

def discrete_mode(image_list,label_list,prior,number,test_image_list,test_label_list):
    lst = [[[0 for k in range(32)] for j in range(784)] for i in range(10)]
    min_bin = [[1 for j in range(784)]for i in range(10)]

    for i in range(len(label_list)):
        class_now = label_list[i]
        for j in range(784) :
            lst[class_now][j][image_list[i][j]]+=1

    for i in range(10):
        for j in range(len(image_list[0])):
            for k in range(32):
                lst[i][j][k] = lst[i][j][k] / number[i]
                if min_bin[i][j]>=lst[i][j][k] and lst[i][j][k]!=0:
                    min_bin[i][j] = lst[i][j][k]


    Postirior = [[0 for j in range(10)] for i in range(len(test_label_list))]
    Postirior_total_num = 0

    data=open("output.txt",'w+') 
    error=0
    for x in range(len(test_label_list)):
  
        Postirior_total_num=0
        min_posterior = 1
        ans=0
        for i in range(10): #caculate every posterior in 0~9
            v=0
            for j in range(len(test_image_list[0])):                
                now_bin = test_image_list[x][j]
                if lst[i][j][now_bin]==0:
                    if min_bin[i][j]==1:
                        v+=math.log10(1/60000) #0.1514
                    else:
                        v+=math.log10(min_bin[i][j]) 
                else:
                    v+=math.log10(lst[i][j][now_bin])

            v+=math.log(prior[i])
            Postirior_total_num +=v
            Postirior[x][i] = v
        
        for i in range(10):
            Postirior[x][i] = Postirior[x][i]/Postirior_total_num
            if Postirior[x][i] < min_posterior:
                min_posterior = Postirior[x][i]
                ans = i

        if ans !=  test_label_list[x] :
            error+=1

        #print data
        data.write("Postirior (in log scale):\n")
        for i in range(len(Postirior[x])):
            data.write(str(i)+":"+str(Postirior[x][i])+"\n")
        data.write("Prediction:"+str(ans)+", Ans:"+str(test_label_list[x])+"\n")
        data.write("\n")
    data.write("Error rate:" +str(error/len(test_label_list)))
    print("Error rate:" +str(error/len(test_label_list)))
    data.close()               
    print_test_image(test_image_list,test_label_list)
    return 0

def continous_mode(image_list,label_list,test_image_list,test_label_list,number,prior):
    SUM = [[0 for j in range(784)]for i in range(10)] #10 number
    SUM_double = [[0 for j in range(784)]for i in range(10)]
    class_now = 0

    G = [[[0 for k in range(2)] for j in range(784)]for i in range(10)] #0 for mean values 1 for variance values
    for i in range(60000):
        for j in range(784):
            SUM[label_list[i]][j]+=image_list[i][j]
            SUM_double[label_list[i]][j]+=(image_list[i][j]**2)

    for i in range(10):# mean
        for j in range(784):
            G[i][j][0] = SUM[i][j]/number[i]

    for i in range(60000):# variance values
        for j in range(784):
            class_now = label_list[i]
            G[class_now][j][1]+=((image_list[i][j]-G[class_now][j][0])**2)

    for i in range(10):
        for j in range(784):
            #G[i][j][1]=G[i][j][1]/number[i] #sol2
            G[i][j][1]=SUM_double[i][j]/number[i]-(G[i][j][0]**2) #sol1

    data=open("output_2.txt",'w+') 
    error = 0
    ans=0
    Postirior_total_num = 0
    Postirior = [[0 for j in range(10)] for i in range(10000)]
    for x in range(10000):
        max_ = 1
        ans = 0
        Postirior_total_num=0
        for i in range(10): #for 0~9 calculate likelihood 
            p=0
            for j in range(784):
                h=0
                v=0
                if G[i][j][1]!=0:
                    h = -(((test_image_list[x][j] - G[i][j][0]) **2)) / (2 * G[i][j][1])
                    v = (0.3989422804 / G[i][j][1]**0.5) * math.exp(h)  
                    if v!=0:
                        p += math.log(v)
                    else:
                        p+=math.log(1/60000) 
                else:
                    if test_image_list[x][j] != G[i][j][0]:
                        p += math.log(1/60000)    
                    else:
                        p+=0 
            p+=math.log(prior[i])
            Postirior_total_num +=p
            Postirior[x][i] = p

        for i in range(10):
            Postirior[x][i] = Postirior[x][i]/Postirior_total_num
            if Postirior[x][i] < max_ :
                max_ = Postirior[x][i]
                ans  = i

        if ans != test_label_list[x]:
            error+=1
            
        data.write("Postirior (in log scale):\n")

        for i in range(len(Postirior[x])):
            data.write(str(i)+":"+str(Postirior[x][i])+"\n")

        data.write("Prediction:"+str(ans)+", Ans:"+str(test_label_list[x])+"\n")
        data.write("\n")   

    print("Error rate:" +str(error/len(test_label_list)))
    data.write("Error rate:" +str(error/len(test_label_list)))
    data.close()  
    print_test_image(test_image_list,test_label_list)
    return 0 

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

def prior_probability_for_class(labal_list): # probability for 0~9
    prior = [0,0,0,0,0,0,0,0,0,0]
    number = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(labal_list)):
        number[labal_list[i]]+=1
    for i in range(len(prior)):
        prior[i] = number[i] / len(labal_list)
    return prior,number 


def main():
    toggle = 0
    toggle=input('enter toggle:')
    
    #train data
    t_i_f = gzip.open("train-images-idx3-ubyte.gz", "rb")
    train_image_list,train_image_list_for_discrete = load_image_data(t_i_f)
    t_l_f = gzip.open("train-labels-idx1-ubyte.gz", "rb")
    train_label_list = load_label_data(t_l_f)
    #test data
    t_i_f = gzip.open("t10k-images-idx3-ubyte.gz", "rb")
    test_image_list,test_image_list_for_discrete = load_image_data(t_i_f)
    t_l_f = gzip.open("t10k-labels-idx1-ubyte.gz", "rb")
    test_label_list = load_label_data(t_l_f)

    prior_for_class,number_for_class = prior_probability_for_class(train_label_list)
    
    if toggle == "0":
        print("toggle = 0")
        discrete_mode(train_image_list_for_discrete,train_label_list,prior_for_class,number_for_class,test_image_list_for_discrete,test_label_list)
    elif toggle == "1":
        print("toggle = 1")
        continous_mode(train_image_list_for_discrete,train_label_list,test_image_list_for_discrete,test_label_list,number_for_class,prior_for_class)


if __name__ == '__main__':
    main()