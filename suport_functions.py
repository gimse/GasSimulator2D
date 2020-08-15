# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:56:53 2020

@author: skitr
"""
import numpy as np
import scipy.sparse as sp

def getDxDy(A):
    n=np.shape(A)[0]
    
    
    blocks_of_matrixes = [[None for x in range(n)] for y in range(n)]
    
    for i in range(n):
            blocks_of_matrixes[i][i]=A;
    Dx=sp.bmat(blocks_of_matrixes);
    
    blocks_of_matrixes = [[None for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            blocks_of_matrixes[i][j]=sp.eye(n,n)*A[i,j];
    Dy=sp.bmat(blocks_of_matrixes)
    
    return (Dx, Dy)