import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784")
mnist.data.shape
'''
MNIST data is in grey scale [0, 255].
Convert it to a binary scale using a threshold of 128.
'''
mnist3 = (mnist.data/128).astype('int')
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
    #compute prob(x/mean)
    # prob[i, k] for ith data point, and kth cluster/mixture distribution
    prob = np.zeros((N, K))
    
    for i in range(N):
        for k in range(K):
            prob[i,k] = np.prod((means[k]**data[i])*((1-means[k])**(1-data[i])))
    
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
    prob = prob*weights
    
    #step 3
    # calcualte the denominator of the resp.s
    row_sums = prob.sum(axis=1)[:, np.newaxis]
    
    # step 4
    # calculate the resp.s
    try:
        prob = prob/row_sums
        return prob
    except ZeroDivisionError:
        print("Division by zero occured in reponsibility calculations!")
            
    

def bernoulliMStep(data, resp):
    '''Re-estimate the parameters using the current responsibilities
    data = N X D matrix
    resp = N X K matrix
    return revised weights (K vector) and means (K X D matrix)
    '''
    N = len(data)
    D = len(data[0])
    K = len(resp[0])
    
    Nk = np.sum(resp, axis=0)
    mus = np.empty((K,D))
    
    for k in range(K):
            mus[k] = np.sum(resp[:,k][:,np.newaxis]*data,axis=0) #sum is over N data points
            try:
                mus[k] = mus[k]/Nk[k]   
            except ZeroDivisionError:
                print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
                break           
    
    return (Nk/N, mus)

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
            try:
                temp1 = ((means[k]**data[i])*((1-means[k])**(1-data[i])))
                temp1 = np.log(temp1.clip(min=1e-50))
                
            except:
                print("Problem computing log(probability)")
            sumK += resp[i, k]*(np.log(weights[k])+np.sum(temp1))
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

from sklearn.utils import shuffle

def pickData(digits, N):
    sData, sTarget = shuffle(mnist3, mnist.target, random_state=30)
    returnData = np.array([sData[i] for i in range(len(sData)) if sTarget[i] in digits])
    return shuffle(returnData, n_samples=N, random_state=30)

def experiments(digits,iters=50):
    '''
    Picks N random points of the selected 'digits' from MNIST data set and
    fits a model using Mixture of Bernoulli distributions.
    And returns the weights and means.
    '''
    
    expData = pickData(digits, N)
    
    D = len(expData[0])

    initWts = np.random.uniform(.25,.75,K)
    tot = np.sum(initWts)
    initWts = initWts/tot
    
    #initMeans = np.random.rand(10,D)
    initMeans = np.full((K, D), 1.0/K)

    return mixOfBernoulliEM(expData, initWts, initMeans, maxiters=iters, relgap=1e-15)

def main():

    #read data
    t_i_f = gzip.open("train-images-idx3-ubyte.gz", "rb")
    train_image_list,image_list_em = load_image_data(t_i_f)
    t_l_f = gzip.open("train-labels-idx1-ubyte.gz", "rb")
    train_label_list = load_label_data(t_l_f)
    #prior,number = prior_probability_for_class(train_label_list)
    #em_ex(image_list_em,train_label_list,number)
    experiments(image_list_em)
 
if __name__ == '__main__':
    main()