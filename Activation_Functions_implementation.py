# -*- coding: utf-8 -*-
"""
Spyder Editor

This .py shows how we can draw different Activation Functions.
More infomation could be found in https://github.com/omarnj-lab/DL.git
"""
print("I am omar")

# Step #1 IMPORTS 
import matplotlib.pyplot as plt
import numpy as np
import math

# Step #2 Define the Activation Functions
def sigmoid(x):
    s =1/(1+np.exp(-x))
    ItsDerivative= s*(1-s)  
    return s, ItsDerivative

x=np.arange(-5,5,0.01)
sigmoid(x)

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    ItsDerivative= 1-t**2
    return t,ItsDerivative

z=np.arange(-4,4,0.01)
tanh(z)[0].size,tanh(z)[1].size

def relu(r) :
  return  max(r, 0)

def der_relu(r):
    if r <= 0:
        return 0
    if r > 0 :
        return 1
    
r = np.arange(-5,5,0.01)

def leaky_relu(l):
    return max(0.01*l,l)

def der_leaky_relu(l):
    if l < 0 :
        return 0.01
    if l >= 0 :
        return 1
    
l = np.arange(-5,5,0.01)

def elu(e):
    if e > 0 :
        return e
    else : 
        return (np.exp(e)-1)

def der_elu(e):
    if e > 0 :
        return 1
    else :
        return np.exp(e)
    
e = np.arange(-5,5,0.01)


def softplus(sp):
    return np.log(1+np.exp(sp))

def der_softplus(sp):
    return 1/(1+np.exp(sp))*np.exp(sp)

sp = np.arange(-5,5,0.01)

def softsign(ss):
    return ss/(1 + abs(ss))

def der_softsign(ss):
    f = 1 + abs(ss)
    return 1/pow(f,2)

ss = np.arange(-4,4,0.01)

def swish(sw):
    return sw/(1 + math.exp(-sw))
def der_swish(sw):
      return (math.exp(sw) * (math.exp(sw) + sw + 1)) / pow((math.exp(sw) + 1) , 2)
 

sw = np.arange(-4,4,0.01)

def sinç(sç):
    return np.sin(sç)/sç
def der_sinç(sç):
      return np.cos(sç)/sç - np.sin(sç)/pow(sç , 2)
 

sç = np.arange(-10,10,0.01)

def logofsigmoid(logs):
    return np.log(1/(1+math.exp(-logs)))
def der_logofsigmoid(logs):
      return 1/(math.exp(logs) + 1)
 

logs = np.arange(-10,10,0.01)

def mish(m):
    return m*(math.tanh(np.log(1 + math.exp(m))))
                
m = np.arange(-10,10,0.01)

def binaryStep(bs):
    return np.heaviside(bs,1)

bs = np.arange(-10, 10)


# Step #3 Create and show plot

plt.plot(x,sigmoid(x)[0], color="Red", linewidth=3, label="sigmoid")
plt.plot(x,sigmoid(x)[1], color="Black", linewidth=3, label="derivative")
plt.legend(loc="upper left")
plt.title("Sigmoid")

plt.show()

plt.plot(z,tanh(z)[0], color="Red", linewidth=3, label="tanh")
plt.plot(z,tanh(z)[1], color="Black", linewidth=3, label="derivative")
plt.legend(loc="upper left")
plt.title("tanh")
plt.show()

plt.plot(r, list(map(lambda r: relu(r),r)),color="Red",linewidth=3, label="relu")
plt.plot(r, list(map(lambda r: der_relu(r),r)), color="Black",linewidth=3,  label="derivative")
plt.title("ReLU")
plt.legend(loc="upper left")
plt.show()

plt.plot(l, list(map(lambda l: leaky_relu(l),l)),color="Red", linewidth=3, label="leaky-relu")
plt.plot(l, list(map(lambda l: der_leaky_relu(l),l)),color="Black", linewidth=3, label="derivative")
plt.title("Leaky-ReLU")
plt.legend(loc="upper left")
plt.show()

plt.plot(e, list(map(lambda e: elu(e),e)),color="Red",  linewidth=3, label="Elu")
plt.plot(e, list(map(lambda e: der_elu(e),e)), color="Black", linewidth=3,label="derivative")
plt.title("ELU")
plt.legend(loc="upper left")
plt.show()



plt.plot(sp, list(map(lambda sp: softplus(sp),sp)), color="Red", linewidth=3, label="softplus")
plt.plot(sp, list(map(lambda sp: der_softplus(sp),sp)),color="Black", linewidth=3, label="derivative")
plt.title("Softplus")
plt.legend(loc="upper left")
plt.show()



plt.plot(ss, list(map(lambda ss: softsign(ss),ss)), color="Red", linewidth=3, label="softsign")
plt.plot(ss, list(map(lambda ss: der_softsign(ss),ss)),color="Black", linewidth=3, label="derivative")
plt.title("softsign")
plt.legend(loc="upper left")
plt.show()

plt.plot(sw, list(map(lambda sw: swish(sw),sw)), color="Red", linewidth=3, label="swish")
plt.plot(sw, list(map(lambda sw: der_swish(sw),sw)),color="Black", linewidth=3, label="derivative")
plt.title("swish")
plt.legend(loc="upper left")
plt.show()


plt.plot(sç, list(map(lambda sç: sinç(sç),sç)), color="Red", linewidth=3, label="sinç")
plt.plot(sç, list(map(lambda sç: der_sinç(sç),sç)),color="Black", linewidth=3, label="derivative")
plt.title("sinç")
plt.legend(loc="upper left")
plt.show()


plt.plot(logs, list(map(lambda logs: logofsigmoid(logs),logs)), color="Red", linewidth=3, label="logofsigmoid")
plt.plot(logs, list(map(lambda logs: der_logofsigmoid(logs),logs)),color="Black", linewidth=3, label="derivative")
plt.title("logofsigmoid")
plt.legend(loc="upper left")
plt.show()



plt.plot(m, list(map(lambda m: mish(m),m)), color="Red", linewidth=3, label="mish")
plt.title("mish")
plt.legend(loc="upper left")
plt.show()


plt.plot(bs, binaryStep(bs) ,  color="Red", linewidth=3, label="step function")
plt.axis('tight')
plt.title("Step Fuction")
plt.legend(loc="upper left")
plt.show()