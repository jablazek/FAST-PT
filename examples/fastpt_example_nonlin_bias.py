"""An example script to run FASTPT
Initializes and calculates all quantities supported by FASTPT
Makes a plot for P_22 + P_13
"""

from time import time
import matplotlib
matplotlib.use('Agg')
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../')
import fastpt as fpt
from fastpt import FASTPT

#Version check
print('This is FAST-PT version', fpt.__version__)

# load the data file

d=np.loadtxt('Pk_test.dat')
# declare k and the power spectrum
k=d[:,0]; P=d[:,1]

# set the parameters for the power spectrum window and
# Fourier coefficient window
#P_window=np.array([.2,.2])
C_window=.75
P_window=None
#document this better in the user manual

# padding length
n_pad=int(0.5*len(k))
to_do=['dd_bias']

# initialize the FASTPT class
# including extrapolation to higher and lower k
# time the operation
t1=time()
fastpt=FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)
t2=time()

# calculate 1loop SPT including bias terms
#P_spt=fastpt.one_loop_dd(P,C_window=C_window)

Growth = 1.0 #placeholder for Growth factor.

PXXNL_b1b2bsb3nl = fastpt.one_loop_dd_bias_b3nl(P, C_window=.75)
Pkz = Growth**2*P
Plinfpt = Growth**2*PXXNL_b1b2bsb3nl[1]
one_loopkz = Growth**4 * PXXNL_b1b2bsb3nl[0]
[Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, sig4kz] = [
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[2]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[3]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[4]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[5]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[6]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[7]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[8] * np.ones_like(
                    PXXNL_b1b2bsb3nl[0]))]
knl2mat = np.tile(k ** 2, (Pkz.shape[0], 1))
k2Pnl1 = np.multiply(knl2mat, Pkz)
sigma=10.0
k2Pnl1_reg = k2Pnl1*(np.exp(-1. * (k ** 2) / sigma ** 2))

t3=time()
print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')

# somewhat representative bias values
b1 = 2.0
b2 = 0.9*(b1-1)**2-0.5 #(this is a numerical fit to simulation k2Pnl1_regdata, but a relationship of this form is motivated in the spherical collapse picture
bs = (-4./7)*(b1-1)
b3nl = (b1-1)
bk=-1.0

Pggsub = (b1 ** 2 * (Pkz+one_loopkz) + b1 * b2 * Pd1d2 + (1. / 4) * b2 * b2 * (Pd2d2 - 2. * sig4kz) + b1 * bs * Pd1s2 +
                  (1. / 2) * b2 * bs * (Pd2s2 - 4. / 3 * sig4kz) + (1. / 4) * bs * bs * (Ps2s2 - 8. / 9 * sig4kz) + (b1 * b3nl ) * sig3nl + 2 *b1 * bk  * k2Pnl1_reg)

Pgg = (b1 ** 2 * (Pkz+one_loopkz) + b1 * b2 * Pd1d2 + (1. / 4) * b2 * b2 * (Pd2d2) + b1 * bs * Pd1s2 +
               (1. / 2) * b2 * bs * (Pd2s2) + (1. / 4) * bs * bs * (Ps2s2) + (b1 * b3nl ) * sig3nl + 2 *b1 * bk  * k2Pnl1_reg )

Pmg = b1 * (Pkz+one_loopkz) + (1. / 2) * b2 * Pd1d2 + (1. / 2) * bs * Pd1s2 + (1. / 2) * b3nl  * sig3nl +  bk  * k2Pnl1_reg

# make a plot of 1loop SPT results

ax=plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$P(k)$', size=20)
ax.set_xlabel(r'$k$', size=20)

ax.plot(k,P,label='linear')
ax.plot(k,one_loopkz, label=r'$P_{22}(k) + P_{13}(k)$' )
ax.plot(k,Pggsub[0],label='gg')
ax.plot(k,Pmg[0],label='mg')

plt.legend(loc=3)
plt.grid()
plt.tight_layout()
plt.savefig('test_Pk.png')
# plt.show()


