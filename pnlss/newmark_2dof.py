#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, spectrogram

from pyvib.forcing import multisine
from pyvib.newmark import Newmark
from pyvib.nonlinear_elements_newmark import NLS, Polynomial

""" Parameters for 2DOF duffing system

    #              x1                          x2                        #
    #            +-->                        +-->                        #
    #            |                           |                           #
    #  d1 __     +--------------+  d2 __     +--------------+  d3 __     #
    #-----__|----|              |-----__|----|              |-----__|----#
    #  k1        |              |  k3        |              |  k4        #
    #__/\  /\  __|      M1      |__/\  /\  __|       M2     |__/\  /\  __#
    #    \/  \/  |              |    \/  \/  |              |    \/  \/  #
    #  k2 ^      |              |            |              | k2  ^      #
    #__/\/ /\  __|              |            |              |__/\/ /\  __#
    #   /\/  \/  +--------------+            +--------------+   /\/  \/  #

Mode & Frequency (rad/s) & Damping ratio (%)
1    & 1.00              &       5.00
2    & 3.32              &       1.51
"""

m1 = 1    # kg
m2 = 1
k1 = 1    # N/m
k2 = 5
k3 = 1
c1 = 0.1  # N/ms
c2 = 0.1
mu1 = 1    # N/m^3
mu2 = 1    # N/m^3

M = np.array([[m1, 0], [0, m2]])
C = np.array([[c1, 0], [0, c2]])
K = np.array([[k1+k2, -k2], [-k2, k2+k3]])
M, C, K = np.atleast_2d(M, C, K)
ndof = M.shape[0]

pol1 = Polynomial(w=[1, 0], exp=3, k=mu1)
pol2 = Polynomial(w=[0, 1], exp=3, k=mu2)
nls = NLS([pol1, pol2])
#nls = NLS()

f0 = 1e-4/2/np.pi
f1 = 5/2/np.pi
fs = 10
T = 100
ns = fs * T
t = np.arange(ns)/fs
# t = np.linspace(0, T, T*fs, endpoint=False)
u = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

u, lines, freq = multisine(f0, f1, N=1024, fs=fs, R=4, P=4)
ns = 1024*4*4
t = np.arange(ns)/fs
fdof = 0
fext = np.zeros((ns, ndof))
fext[:, fdof] = u.ravel()

sys = Newmark(M, C, K, nls)
dt = t[1] - t[0]
x, xd, xdd = sys.integrate(fext, dt, x0=0, v0=0, sensitivity=False)


plt.figure()
plt.plot(t, x, '-k', label=r'$x_1$')
#plt.plot(t, x, '-r', label=r'$x_2$')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')
plt.title('Force type: {}, periods:{:d}')
plt.legend()

plt.figure()
plt.plot(np.abs(np.fft.fft(x[6*1024:7*1024, 0])))

# plot force and spectogram
# plt.plot(t,u)
# ff, tt, Sxx = spectrogram(u, fs=fs, noverlap=2, nperseg=10,
#                         nfft=56)
#plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
#plt.title('Logarithmic Chirp, f(0)=1500, f(10)=250')
#plt.xlabel('t (sec)')
#plt.ylabel('Frequency (Hz)')
# plt.grid()


plt.show()
