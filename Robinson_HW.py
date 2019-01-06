#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:49:38 2018

@author: Rodrigo Castro
"""

def main():

    import numpy as np
    import matplotlib.pyplot as plt
    
    x=3
    y=1
    alpha=2
    
    theta=np.zeros((6,1))
    theta[0]=1
    theta[1]=-2
    theta[2]=-0.5
    theta[3]=1
    theta[4]=-2
    theta[5]=3
    
    #theta=np.random.randn(6,1)
    
    def feedforward(x,y,theta):
        z1=np.tanh(theta[0]*x+theta[1])
        z2=np.tanh(theta[2]*x+theta[3])
        yhat=1/(1+np.exp(-(theta[4]*z1+theta[5]*z2)))
        J=1/2*(yhat-y)**2
        
        return z1,z2,yhat,J
    
    def backprop(z1,z2,yhat,J):
        
        dJ=np.zeros((6,1))
        
        dJ_dyhat=yhat-y
        dyhat_dz1=yhat*(1-yhat)*theta[4]
        dyhat_dz2=yhat*(1-yhat)*theta[5]
        dyhat_dtheta4=yhat*(1-yhat)*z1
        dyhat_dtheta5=yhat*(1-yhat)*z2
        dz1_dtheta0=(1-z1**2)*x
        dz1_dtheta1=(1-z1**2)
        dz2_dtheta2=(1-z2**2)*x
        dz2_dtheta3=(1-z2**2)
    
        dJ[0]=dJ_dyhat*dyhat_dz1*dz1_dtheta0
        dJ[1]=dJ_dyhat*dyhat_dz1*dz1_dtheta1
        dJ[2]=dJ_dyhat*dyhat_dz2*dz2_dtheta2
        dJ[3]=dJ_dyhat*dyhat_dz2*dz2_dtheta3
        dJ[4]=dJ_dyhat*dyhat_dtheta4
        dJ[5]=dJ_dyhat*dyhat_dtheta5
          
        return dJ
    
    J_values=[]
    n=5
    for i in range(0,n):    
        print('theta='+str(theta))
        z1,z2,yhat,J=feedforward(x,y,theta)
        dJ=backprop(z1,z2,yhat,J)    
        theta-=alpha*dJ
        J_values.append(J)
        
        print('J'+str(J))
        
    plt.plot(J_values,'ro')
    plt.xlabel('iterations')
    vert_label=plt.ylabel('J')
    vert_label.set_rotation(0)
    print('The model prediction after '+str(n)+' iterations is '+str(yhat))
    
main()