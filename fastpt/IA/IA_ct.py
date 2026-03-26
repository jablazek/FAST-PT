from __future__ import division
import numpy as np
from ..utils.J_table import J_table
from ..utils.J_k import J_k
import sys
from time import time
from numpy import log, exp, pi
from scipy.signal import fftconvolve as convolve

def IA_tij_feG2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_feG2=np.array([[0,0,0,2,0,13/21],\
            [0,0,0,2,2,8/21],\
            [1,-1,0,2,1,1/2],\
            [-1,1,0,2,1,1/2]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_feG2.shape[0]):
        x=J_table(l_mat_tij_feG2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_heG2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_heG2=np.array([[0,0,0,0,0,-9/70],\
            [0,0,2,0,0,-26/63],\
            [0,0,0,0,2,-15/49],\
            [0,0,2,0,2,-16/63],\
            [0,0,1,1,1,81/70],\
            [0,0,1,1,3,12/35],\
            [0,0,0,0,4,-16/245],\
            [1,-1,0,0,1,-3/10],\
            [1,-1,2,0,1,-1/3],\
            [1,-1,1,1,0,1/2],\
            [1,-1,1,1,2,1],\
            [1,-1,0,2,1,-1/3],\
            [1,-1,0,0,3,-1/5]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_heG2.shape[0]):
        x=J_table(l_mat_tij_heG2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_F2F2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_F2F2=np.array([[0,0,0,0,0,1219/1470],\
            [0,0,0,0,2,671/1029],\
            [0,0,0,0,4,32/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,62/35],\
            [1,-1,0,0,4,8/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_F2F2.shape[0]):
        x=J_table(l_mat_tij_F2F2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_G2G2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_G2G2=np.array([[0,0,0,0,0,851/1470],\
            [0,0,0,0,2,871/1029],\
            [0,0,0,0,4,128/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,54/35],\
            [1,-1,0,0,4,16/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_G2G2.shape[0]):
        x=J_table(l_mat_tij_G2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_F2G2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_F2G2=np.array([[0,0,0,0,0,1003/1470],\
            [0,0,0,0,2,803/1029],\
            [0,0,0,0,4,64/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,58/35],\
            [1,-1,0,0,4,12/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_F2G2.shape[0]):
        x=J_table(l_mat_tij_F2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def P_IA_13G(k,P):
    N=k.size
    n = np.arange(-N+1,N )
    dL=log(k[1])-log(k[0])
    s=n*dL
    cut=7
    high_s=s[s > cut]
    low_s=s[s < -cut]
    mid_high_s=s[ (s <= cut) &  (s > 0)]
    mid_low_s=s[ (s >= -cut) &  (s < 0)]

    # For Zbar
    Z1=lambda r : (12/r**2-26+4*r**2-6*r**4+(3/r**3)*(r**2-1)**3*(r**2+2)*log((r+1.)/np.absolute(r-1.))) *r
    Z1_low=lambda r : (-224/5+1248/(35*r**2)-608/(105*r**4)-26/(21*r**6)-4/(35*r**8)+34/(21*r**10) -4/(3*r**12)) *r
    Z1_high=lambda r: ((-32*r**2)/5-(96*r**4)/7+(352*r**6)/105+(164*r**8)/105-(58*r**10)/35+(4*r**12)/21+(2*r**14)/3) *r

    f_mid_low=Z1(exp(-mid_low_s))
    f_mid_high=Z1(exp(-mid_high_s))
    f_high = Z1_high(exp(-high_s))
    f_low = Z1_low(exp(-low_s))

    f=np.hstack((f_low,f_mid_low,-16.,f_mid_high,f_high))
    # print(f)

    g= convolve(P, f) * dL
    g_k=g[N-1:2*N-1]
    deltaE2= 1/84 * k**3/(2*pi)**2 * P*g_k
    return deltaE2

def P_22F_reg(k,P, P_window, C_window, n_pad):
    # P_22 Legendre components
    # We calculate a regularized version of P_22
    # by omitting the J_{2,-2,0} term so that the
    # integral converges.  In the final power spectrum
    # we add the asymptotic portions of P_22 and P_13 so
    # that we get a convergent integral.  See section of XXX.

    param_matrix=np.array([[0,0,0,0],[0,0,2,0],[0,0,4,0],[2,-2,2,0],\
                            [1,-1,1,0],[1,-1,3,0],[2,-2,0,1] ])


    Power, mat=J_k(k,P,param_matrix, P_window=P_window, C_window=C_window, n_pad=n_pad)
    A=1219/1470.*mat[0,:]
    B=671/1029.*mat[1,:]
    C=32/1715.*mat[2,:]
    D=1/3.*mat[3,:]
    E=62/35.*mat[4,:]
    F=8/35.*mat[5,:]
    reg=1/3.*mat[6,:]

    return 2*(A+B+C+D+E+F)+ reg

def P_22G_reg(k,P, P_window, C_window, n_pad):
    # P_22 Legendre components
    # We calculate a regularized version of P_22
    # by omitting the J_{2,-2,0} term so that the
    # integral converges.  In the final power spectrum
    # we add the asymptotic portions of P_22 and P_13 so
    # that we get a convergent integral.  See section of XXX.

    param_matrix=np.array([[0,0,0,0],[0,0,2,0],[0,0,4,0],[2,-2,2,0],\
                            [1,-1,1,0],[1,-1,3,0],[2,-2,0,1]])


    Power, mat=J_k(k,P,param_matrix, P_window=P_window, C_window=C_window, n_pad=n_pad)
    A=1003/1470.*mat[0,:]
    B=803/1029.*mat[1,:]
    C=64/1715.*mat[2,:]
    D=1/3.*mat[3,:]
    E=58/35.*mat[4,:]
    F=12/35.*mat[5,:]
    reg=1/3.*mat[6,:]
     

    return 2*(A+B+C+D+E+F)+ reg

def IA_tij_F2G2reg():
    # P_22 Legendre components
    # We calculate a regularized version of P_22
    # by omitting the J_{2,-2,0} term so that the
    # integral converges.  In the final power spectrum
    # we add the asymptotic portions of P_22 and P_13 so
    # that we get a convergent integral.  See section of XXX.
     
    l_mat_tij_F2G2=np.array([[0,0,0,0,0,1003/1470],\
            [0,0,0,0,2,803/1029],\
            [0,0,0,0,4,64/1715],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,58/35],\
            [1,-1,0,0,4,12/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_F2G2.shape[0]):
        x=J_table(l_mat_tij_F2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]


def P_IA_13F(k,P):
    N=k.size
    n= np.arange(-N+1,N )
    dL=log(k[1])-log(k[0])
    s=n*dL

    cut=7
    high_s=s[s > cut]
    low_s=s[s < -cut]
    mid_high_s=s[ (s <= cut) &  (s > 0)]
    mid_low_s=s[ (s >= -cut) &  (s < 0)]

    Z=lambda r : (12./r**2 +10. + 100.*r**2-42.*r**4 \
    + 3./r**3*(r**2-1.)**3*(7*r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
    Z_low=lambda r : (352./5.+96./.5/r**2 -160./21./r**4 - 526./105./r**6 +236./35./r**8-50./21./r**10-4./3./r**12) *r
    Z_high=lambda r: (928./5.*r**2 - 4512./35.*r**4 +416./21.*r**6 +356./105.*r**8+74./35.*r**10-20./3.*r**12+14./3.*r**14) *r

    f_mid_low=Z(exp(-mid_low_s))
    f_mid_high=Z(exp(-mid_high_s))
    f_high = Z_high(exp(-high_s))
    f_low = Z_low(exp(-low_s))

    f=np.hstack((f_low,f_mid_low,80,f_mid_high,f_high))

    g= convolve(P, f) * dL
    g_k=g[N-1:2*N-1]
    P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

    return P_bar
