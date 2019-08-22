#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pyvib.pnlss import PNLSS
from pyvib.signal import Signal

""" Hammerstein system with zero initial condition. """

np.random.seed(10)
N = int(2e3)  # Number of samples
u = np.random.randn(N)


def f_NL(x): return x + 0.2*x**2 + 0.1*x**3


b, a = signal.cheby1(2, 5, 2*0.3, 'low', analog=False)
x = f_NL(u)  # Intermediate signal
y = signal.lfilter(b, a, x)  # output
scale = np.linalg.lstsq(u[:, None], x)[0].item()
# Initial linear model = scale factor times underlying dynamics
sys = signal.tf2ss(scale*b, a)
T1 = 0  # No periodic transient
T2 = 200  # number of transient samples to discard
sig = Signal(u[:, None, None, None], y[:, None, None, None])
sig.average()
model = PNLSS(sys, sig)

# Hammerstein system only has nonlinear terms in the input.
# Quadratic and cubic terms in state- and output equation
model.nlterms('x', [2, 3], 'inputsonly')
model.nlterms('y', [2, 3], 'inputsonly')
model.transient(T1=T1, T2=T2)
model.optimize(weight=False, nmax=50)
yopt = model.simulate(u)[1]


def rmse(y, yhat): return np.sqrt(np.mean((y-yhat.T)**2))


print(f'RMS error {rmse(y, yopt)}')

t = np.arange(N)
plt.ion()
plt.figure()
plt.plot(t, y)
plt.plot(t, y-yopt, '--k', lw=3)
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend(['True', 'Error'])
plt.show()
