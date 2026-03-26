from __future__ import division
import numpy as np
from ..utils.J_table import J_table 
import sys
from time import time
from numpy import log, exp, pi
from scipy.signal import fftconvolve as convolve

def P_IA_13S2F2(k,P):

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL
	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]



	Z1=lambda r : (((4.*r*(45.-165.*r**2+379.*r**4+45.*r**6)+45.*(-1.+r**2)**4*log((-1.+r)**2)-90.*(-1.+r**2)**4*log(np.absolute(1.+r)))/(2016.*r**3))-68./63*r**2)/2.
	Z1_high=lambda r : ((-16*r**2)/21. + (16*r**4)/49. - (16*r**6)/441. - (16*r**8)/4851. - 16*r**10/21021.- 16*r**12/63063.)/2.
	Z1_low=lambda r: (-16/21.+ 16/(49.*r**2) - 16/(441.*r**4) - 16/(4851.*r**6) - 16/(21021.*r**8) - 16/(63063.*r**10)- 16/(153153.*r**12)  )/2.


	f_mid_low=Z1(exp(-mid_low_s))*exp(-mid_low_s)
	f_mid_high=Z1(exp(-mid_high_s))*exp(-mid_high_s)
	f_high = Z1_high(exp(-high_s))*exp(-high_s)
	f_low = Z1_low(exp(-low_s))*exp(-low_s)

	f=np.hstack((f_low,f_mid_low,-0.2381002916036672,f_mid_high,f_high))


	g= convolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	S2F2= k**3/(2.*pi**2) * P*g_k
	return S2F2


def IA_gb2_fe():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_fe=np.array([[0,0,2,0,0,1]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_fe.shape[0]):
        x=J_table(l_mat_gb2_fe[i])
        table=np.row_stack((table,x))
    return table[1:,:]


def IA_gb2_he():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_he=np.array([[0,0,0,0,0,-1/6],\
            [0,0,2,0,0,-1/3],\
            [0,0,0,0,2,-1/3],\
            [0,0,1,1,1,3/2],\
            [0,0,2,0,0,-1/3]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_he.shape[0]):
        x=J_table(l_mat_gb2_he[i])
        table=np.row_stack((table,x))
    return table[1:,:]

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

def IA_gb2_S2fe():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_S2fe=np.array([[0,0,2,0,2,2/3]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_S2fe.shape[0]):
        x=J_table(l_mat_gb2_S2fe[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_gb2_S2he():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_gb2_S2he=np.array([[0,0,0,0,0,-2/45],\
            [0,0,0,0,2,-11/63],\
            [0,0,2,0,2,-2/9],\
            [0,0,0,2,2,-2/9],\
            [0,0,1,1,1,2/5],\
            [0,0,1,1,3,3/5],\
            [0,0,0,0,4,-4/35]],dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_gb2_S2he.shape[0]):
        x=J_table(l_mat_gb2_S2he[i])
        table=np.row_stack((table,x))
    return table[1:,:]