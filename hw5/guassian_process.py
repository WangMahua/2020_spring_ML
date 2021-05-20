#! /usr/bin/env python3
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5,"alpha_": 0.01}
        self.sigma_ = 0.2
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # def kernel_for_optimize(l_,a_, x1, x2):
        #     m, n = x1.shape[0],x2.shape[0]
        #     dist_matrix = np.zeros((m,n),dtype = float)
        #     dist_matrix_A = []
        #     for i in range(m):
        #         for j in range(n):
        #             dist_matrix[i][j] = np.sum((x1[i]-x2[j])**2)
        #     #dist_matrix_A = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        #     return self.sigma_ ** 2 * (1 + dist_matrix / (2*a_*l_))**(-a_)

        # def negative_log_likelihood_loss(l_,a_):
        #     # self.params["l"],self.params["alpha_"]= params[0], params[1], 
        #     C = self.train_X 
        #     Kyy = kernel_for_optimize(l_ , a_, C, C) 
        #     Kyy +=1e-8 * np.eye(len(self.train_X))
        #     A = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
        #     print(A.shape)
        #     return A

        # if self.optimize:
        #     l_ = 0.5
        #     a_ = 0.1
        #     x0 = [l_,a_]
        #     res = minimize(negative_log_likelihood_loss(l_,a_), x0,
        #            bounds=((1e-4, 1e4),(1e-4, 1e4)),jac = 'gradient',
        #            method='TNC')

        #     self.params["l"],self.params["alpha_"] = res.x[0], res.x[1]
        #     print(res.fun)
        #     print(res.success)
        #     print(res.x)
 
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        m, n = x1.shape[0],x2.shape[0]
        dist_matrix = np.zeros((m,n),dtype = float)
        dist_matrix_A = []
        for i in range(m):
            for j in range(n):
                dist_matrix[i][j] = np.sum((x1[i]-x2[j])**2)
        #dist_matrix_A = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma_ ** 2 * (1 + dist_matrix / (2*self.params["alpha_"]*self.params["l"]))**(-self.params["alpha_"])




def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()



def main():
 
    #read data
    f = open("data/input.data")
    #build data
    data=[]
    X=[]
    Y=[]
    for line in f:
        data_x=[]
        data_y=[]
        currentline = line.strip().split(" ")
        #data_x.append(1)
        data_x.append(float(currentline[0]))
        data_y.append(float(currentline[1]))
        X.append(data_x) #basic function
        Y.append(data_y) #basic function

    train_X = np.array(X).reshape(-1, 1)
    train_y = np.array(Y)
    test_X = np.arange(-60, 60, 1).reshape(-1, 1)

    gpr = GPR()
    gpr.fit(train_X, train_y)
    mu, cov = gpr.predict(test_X)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))#95% confidence interval of f
    plt.figure()
    plt.title("l=%.2f alpha=%.2f" % (gpr.params["l"], gpr.params["alpha_"]))
    plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    plt.plot(test_X, test_y, label="predict")
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()





    
    
if __name__ == '__main__':
    main()
