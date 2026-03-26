from __future__ import division
import numpy as np
from ..utils.J_table import J_table 
import sys
from time import time

def IA_gb2_F2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_F2=np.array([[0,0,0,0,0,17/21],\
            [0,0,0,0,2,4/21],\
            [1,-1,0,0,1,1/2],\
            [-1,1,0,0,1,1/2]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_F2.shape[0]):
        x=J_table(l_mat_gb2_F2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_gb2_G2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_G2=np.array([[0,0,0,0,0,13/21],\
            [0,0,0,0,2,8/21],\
            [1,-1,0,0,1,1/2],\
            [-1,1,0,0,1,1/2]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_G2.shape[0]):
        x=J_table(l_mat_gb2_G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_gb2_S2F2():
     # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_S2F2=np.array([[0,0,0,0,0,8/315],\
            [0,0,0,0,2,254/441],\
            [0,0,0,0,4,16/245],\
            [1,-1,0,0,1,2/15],\
            [1,-1,0,0,3,1/5],\
            [-1,1,0,0,1,2/15],\
            [-1,1,0,0,3,1/5]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_S2F2.shape[0]):
        x=J_table(l_mat_gb2_S2F2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_gb2_S2G2():
     # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_S2G2=np.array([[0,0,0,0,0,16/315],\
            [0,0,0,0,2,214/441],\
            [0,0,0,0,4,32/245],\
            [1,-1,0,0,1,2/15],\
            [1,-1,0,0,3,1/5],\
            [-1,1,0,0,1,2/15],\
            [-1,1,0,0,1,1/5]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_S2G2.shape[0]):
        x=J_table(l_mat_gb2_S2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]