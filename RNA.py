Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import random
import math
import copy

def inicRede(Y,A):
  #o vetor Y contem os Y0s e o A contem o numero de neuronios por layer.
  W=[[[random.random() for i in range(len(Y))] for n in range(A[0])]]
  B=[[random.random() for n in range(A[0])]]
  for l in range(1,len(A)):
    B+=[[random.random() for n in range(A[l])]]
    W+=[[[random.random() for i in range(A[l-1])] for n in range(A[l])]]
  return feedforward(W,B,Y)

  
def feedforward(W,B,Y):
  Y=[Y]
  Z=[]
  for l in range(len(B)):
    A=[];Z1=[]
    for n in range(len(W[l])):
      s=0.0
      for j in range(len(W[l][n])):
        s+=Y[l][j]*W[l][n][j]
      z=s-B[l][n]
      Z1+=[z]
      A+=[sigmoid(z)]
    Z+=[Z1]
    Y+=[A]
  return (backprop(0.1,Y,Z,W,B))
  
def backprop(eta,Y,Z,W,B):
  Erro=copy.copy(Z)
  for n in range(len(Z[len(Z)-1])):
    Erro[len(Z)-1][n]=(Y[len(Z)][n]-n)*sigmoid_Der(Z[len(Z)-1][n])
    B[len(Z)-1][n]-=eta*Erro[len(Z)-1][n]
    for i in range(len(Y[len(Z)-1])):
      W[len(Z)-1][n][i]-=eta*Erro[len(Z)-1][n]*Y[len(Z)][i]
  for l in range(len(Z)-2,0,-1):
    for n in range(len(Z[l])):
      erro=0
      for i in range(len(Z[l+1])):
        erro+=W[l+1][i][n]*Erro[l+1][1]*sigmoid_Der(Z[l][n])
      Erro[l][n]=erro
      B[l][n]-=eta*Erro[l][n]
      for i in range(len(Z[l])):
        print(l,n,i,W[1][1][0],Erro[1][1])
        W[l][n][i]-=eta*Erro[l][n]*Y[l][i]
  return (W)
  
def sigmoid(z):
  return 1.0/(1.0+math.exp(-z))
  
def sigmoid_Der(z):
    return sigmoid(z)*(1-sigmoid(z))
  
print(inicRede([1.0,0.0],[3,4,3,10]))
