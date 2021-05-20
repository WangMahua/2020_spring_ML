#! /usr/bin/env python
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import numpy as np


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

def luDecomposition(A, b):
    upper=A
    lower=[[0 for x in range(len(A))]
             for y in range(len(A))]
    #find upper and lower
    for i in range(len(A)-1):
        for k in range(i+1,len(A)):
            temp = upper[k][i]/upper[i][i]
            lower[k][i] = temp
            for m in range(len(upper[0])):
                upper[k][m]=upper[k][m]-temp*upper[i][m]
    for j in range(len(upper)):
        lower[j][j]=1

	#perform substitutioan Ly=b
    y = [0 for i in range(len(lower))]
    for i in range(0,len(A),1):
        y[i] = b[i][0]/lower[i][i]
        for k in range(0,i,1):
            y[i] -= y[k]*lower[i][k]
    
	#perform substitution Ux=y
    x = [0 for i in range(len(upper))]
    for i in range(len(upper)-1,-1,-1):
        for k in range (len(upper)-1,i,-1):
            y[i] -= x[k]*upper[i][k]
        x[i] = y[i]/upper[i][i]

    x_vector=[[0 for row in range(1)] for col in range(len(x))]
    
    for i in range(len(x)):
        x_vector[i][0]=x[i]
        
    return x_vector

def LSE(A,b,lambda_LSE):
    A_t=transposeMatrix(A)
    G=multiplicationMatrix(A_t,A)
    if lambda_LSE != 0:
        for i in range(len(G)):
            G[i][i]+=lambda_LSE
    G_i=getMatrixInverse(G)  
    Atb=multiplicationMatrix(A_t,b)
    x=luDecomposition(G,Atb)
    return x 

def total_error(A,x,b):
    Ax=multiplicationMatrix(A,x)
    Ax_b=[[0 for row in range(len(Ax[0]))] for col in range(len(Ax))]
    lse_marix=[]
    Ax_b=minus_matrix(Ax,b)
    Ax_b_t=transposeMatrix(Ax_b)
    lse_marix=multiplicationMatrix(Ax_b_t,Ax_b)
    lse_error=lse_marix[0][0]
    lambda_matrix=multiplicationMatrix(transposeMatrix(x),x)
    error_sum=lse_error
    
    return error_sum

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

def mul_matrix(A,n):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j]*n
    return A  

def newton_method(A,b):
    x_initial = [[4 for row in range(1)] for col in range(len(A[0]))]
    x = [[0 for row in range(1)] for col in range(len(A[0]))]
    Atb = multiplicationMatrix(transposeMatrix(A),b)
    gradient_f = multiplicationMatrix(multiplicationMatrix(transposeMatrix(A),A),x_initial)
    gradient_f = mul_matrix(minus_matrix(gradient_f,Atb),2)
    hession_function = Ax_b=[[0 for row in range(len(A[0]))] for col in range(len(A))]
    hession_function = multiplicationMatrix(transposeMatrix(A),A)
    hession_function = mul_matrix(hession_function,2)
    hession_function_i=getMatrixInverse(hession_function)
    x = minus_matrix(x_initial,multiplicationMatrix(getMatrixInverse(hession_function),gradient_f))
           
    return x 

def plot_fitting_line(x):
    f_c = []
    c = np.linspace(-6,6,num=100)
    for m in range(len(c)):
        sum = 0
        for n in range(len(x)):
            coef = x[n][0]
            d=len(x)-n-1
            x_d=c[m]**d
            sum += coef*x_d
        f_c.append(sum)
    return c,f_c



def main():
 
    #read data
    f = open("testfile.txt")
    n=input('enter dimension:')
    lambda_=input('enter lambda:')

    #build data
    data=[]
    A=[]
    b=[]
    x_for_plot=[]
    y_for_plot=[]
    for line in f:
        data_x=[]
        data_y=[]
        currentline = line.strip().split(",")
        for i in range(n):
            if i == n-1 :
                data_x.append(1)
            else:
                data_x.append(float(currentline[0])**(n-i-1))
        x_for_plot.append(float(currentline[0]))
        y_for_plot.append(float(currentline[1]))
        A.append(data_x)
        data_y.append(float(currentline[1]))
        b.append(data_y)

    #calculate x 
    x_LSE=LSE(A,b,lambda_)
    x_NM=newton_method(A,b)
    
    #print
    print('LSE:')
    print('Fitting line: ',end='')
    for i in range(len(x_LSE)):
        if i == len(x_LSE)-1 :
            print(x_LSE[i][0])
        else: 
            c=str(x_LSE[i][0])   
            d=str(len(x_LSE)-i-1)        
            print(c+'x^'+d+'+ ',end='')
    print('Total error: ',end='')
    print(total_error(A,x_LSE,b))

    print("Newton's Method:")
    print('Fitting line: ',end='')
    for i in range(len(x_NM)):
        if i == len(x_NM)-1 :
            print(x_NM[i][0])
        else: 
            c=str(x_NM[i][0])   
            d=str(len(x_NM)-i-1)        
            print(c+'x^'+d+'+ ',end='')
    print('Total error: ',end='')
    print(total_error(A,x_NM,b))


    #plot
    c_LSE,f_c_LSE = plot_fitting_line(x_LSE)
    c_NM,f_c_NM = plot_fitting_line(x_NM)

    fig, axs = plt.subplots(2)
    axs[0].plot(c_LSE,f_c_LSE, 'r')
    axs[0].scatter(x_for_plot,y_for_plot, s=40, c='green')
    axs[0].set_xlim([-6,6])
    axs[1].plot(c_NM,f_c_NM, 'r')
    axs[1].scatter(x_for_plot,y_for_plot, s=40, c='green')
    axs[1].set_xlim([-6,6])
    plt.show()


    
    
if __name__ == '__main__':
    main()
