import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z) )

sigmoid_vec = np.vectorize( sigmoid )

def sigmoid_prime(z):
    return sigmoid(z) * ( 1 - sigmoid(z) );

sigmoid_prime_vec = np.vectorize( sigmoid_prime )

def Test_Sigmoid_Vec():
    tmp = np.arange(6).reshape(2,3)
    print tmp
    sig_tmp = sigmoid_vec(tmp)
    print sig_tmp

