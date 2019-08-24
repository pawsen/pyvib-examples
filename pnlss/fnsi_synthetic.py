#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy.linalg import norm

from pyvib.common import db
from pyvib.fnsi import FNSI
from pyvib.frf import covariance
from pyvib.nonlinear_elements import Polynomial
from pyvib.signal import Signal
from pyvib.subspace import Subspace
from pyvib.modal import modal_ac , frf_mkc

system = 'duffing'
system = '2dof'
system = 'nlbeam'

if system == 'duffing':
    fpath = 'data/Duffing_SyntheticData'
    uname = 'u_1'
    yname = 'y_1'
    #uname = 'u_100'
    #yname = 'y_100'
    # ID parameters
    Ptr = 4
    n = 2
    r = 20
    exponent = [3]
    w = [1]
    cr_true = 1e8
elif system == '2dof':
    fpath = 'data/2DOF_SyntheticData'
    uname = 'u_05'
    yname = 'y_05_noise10'
    # ID parameters
    Ptr = 5
    n = 4
    r = 40
    exponent = [3]
    w = [1,0]
    cr_true = 0.5
elif system == 'nlbeam':
    fpath = 'data/NLBeam_SyntheticData⁄'
    uname = 'u_15'
    yname = 'y_15'
    # ID parameters
    Ptr = 5
    n = 6
    r = 40
    exponent = [3,2]
    w = np.zeros(7); w[6] = 1
    cr_true = np.array([8e9, -1.05e7])

datau = sio.loadmat(f'{fpath}/{uname}.mat')
datay = sio.loadmat(f'{fpath}/{yname}.mat')

# load data
dictget = lambda d, *k: [d[i].item() for i in k]
fs, N, P, fmin, fmax, Amp, iu = dictget(datau, 'fs', 'N', 'P', 'fmin', 'fmax', 'Amp', 'iu')
y = datay['y']
u = datau['u']
try:
    lines = datau['flines'].squeeze().astype(int)
except:
    f1 = int(np.floor(fmin/fs * N))
    f2 = int(np.ceil(fmax/fs * N))
    lines = np.arange(f1+1, f2+1)

freq = np.arange(N)/N *fs
# convert to python indexing
iu -= 1


# reshape, we need the format (N,p,R,P)
m, cu = u.shape; u = u.T if m > cu else u
p, cy = y.shape; y = y.T if p > cy else y

# (p, P*N)
p = y.shape[0]
m = u.shape[0]

# For FNSI we always have R = 1
u.shape = (m,P,N)
y.shape = (p,P,N)
u = u.transpose(2,0,1)[:,:,None,Ptr:]
y = y.transpose(2,0,1)[:,:,None,Ptr:]

# start ID
sig = Signal(u,y,fs=fs)
um, ym = sig.average()
sig.lines = lines

nlx = [Polynomial(exponent=ex, w=w) for ex in exponent]
# nlx = None

#n = 4
bd_method = 'nr'
bd_method = 'opt'
# bd_method = 'explicit'
fnsi1 = FNSI()
fnsi1.set_signal(sig)
fnsi1.add_nl(nlx=nlx)
fnsi1.estimate(n=n, r=r, bd_method=bd_method, weight=False)
fnsi1.transient(T1=N)
fnsi2 = deepcopy(fnsi1)
fnsi2.optimize(lamb=100, weight=False, nmax=10, info=1)

if nlx is not None:
    G, knl = fnsi2.nl_coeff(iu)

    cr = knl.real
    cim = knl.imag
    ratio =  np.log10(np.abs(cr.mean(0)/cim.mean(0)));
    print('Ratio of the real and imaginary parts of the nonlinear coefficient (log)');
    print(f'{ratio}')

    print('Error on the nonlinear coefficient (%)')
    print(f'{100*(cr.mean(0)-cr_true)/cr_true}')
    
    for exp, coef in zip(exponent, cr.T):
        plt.figure()
        plt.plot(freq[lines], coef)
        #plt.ylim([0.9e8, 1.1e8])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(f'Real part of the NL coefficient (N/m^{exp})')
    

# subspace ID
sig.bla()
#linmodel1 = Subspace(sig)
#linmodel1.estimate(n=n, r=r, weight=False, bd_method=bd_method)
#linmodel2 = deepcopy(linmodel1)
#linmodel2.optimize(weight=False, info=1)

plt.figure()
plt.plot(freq[lines], db(np.abs(sig.G[:,0,0])))


## DEBUG STUFF

#from pyvib.subspace import bd_nr, output_costfcn, frf_jacobian, jacobian_freq
#from scipy.io import loadmat
#data = loadmat('test.mat')
#U = data['E'].T
#Y = data['Y'].T
#lines = data['flines'].squeeze()
#A = data['A'].astype(float)
#C = data['C'].astype(float)
#N = data['N'].item()
#theta = data['theta'].squeeze().astype(int)
#freq = lines/N
#U = U[lines]
#Y = Y[lines]
##n = 2
#m = 2
#p = 2
#theta = np.arange(1,7)
#theta = np.array([1,3,2,4,5,6])
#theta = np.r_[1+np.arange(2*4).reshape((4,2), order='C').flatten('F'), 
#              4*2+1+np.arange(2*2).reshape((2,2), order='C').flatten('F')]
# theta = np.r_[[1, 3, 5, 7, 2, 4, 6, 8],[ 9, 11, 10, 12]]
#theta = np.arange(12) + 1
#cost = output_costfcn(theta, A, C, n, m, p, freq, U, Y, weight=False)
#jac = frf_jacobian(theta, A, C, n, m, p, freq, U, weight=False)
#B, D = bd_nr(A, C, None, freq, n, r, m, p, U, Y, weight=False)

#z = np.exp(2j*np.pi*freq)
#_, JB, _, JD = jacobian_freq(A, B, C, z)