# coding: utf-8

import pdb
import numpy as np
from scipy.integrate import simps
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy.linalg import eigvals
from scipy.linalg import solve
from scipy.linalg.lapack import dgeev
from scipy.sparse.linalg import eigs

# --*-- Set functions --*--

# Set initial function
def Psi_0(x):
    return x*np.exp(-x**2/2)

# Set potential
def V(x):
    return 25*np.sin(x/5)**2


# --*-- Set constants and variables for initial condition --*--

# Volume, Maximum x-step
L, xN = 20.0, 300

# Interaction constant, Chemical potential
gN, mu = 50, 10

# Maximum time, Time step 
tMax, tN = 7, 1024

# x-step, Time step
h, dt = L/xN, tMax/tN

# Set x-space
x = np.linspace(-L/2, L/2, xN)

# Set time propagator of poptential term except nonlinear term
pre_expV = np.exp(-V(x)*dt)
expK = np.exp(-(2*np.pi*fftfreq(xN, d=1/xN)/L)**2*dt)

# Container of maximum imaginary eigenvalue
imag_igen = []

# Preparation of sub-diagonal of calL
alpha = np.diag(np.array([h**(-2)]*xN))
alpha = np.vstack((np.array([0]*xN), alpha[:xN-1]))
alpha += alpha.T

# --*-- Gross-Pitaevskii & Bogoliubov-de Gennes --*--

# For-loop on gN
arr_gN = np.linspace(0, 30, 60)
for gN in arr_gN:
    # -*- Gross-Pitaevskii equation on symplectic -*-
    
    # Set operators for time propagation for every gNs
    arr_Psi = Psi_0(x)
    expV = np.exp((mu - gN*Psi_0(x)**2)*dt)*pre_expV
    
    # Time evolution
    for i in range(tN):
        # Time evolution
        arr_Psi = ifft(fft(arr_Psi*expV)*expK)

        # Correction of chemical potential mu
        mu -= (simps(np.absolute(arr_Psi)**2, x) - 1)/(2*dt)

        # Normalization of order parameter arr_Psi
        arr_Psi /= np.sqrt(simps(np.absolute(arr_Psi)**2, x))
        
        # Reconfigure expV
        expV = np.exp((mu - gN*np.absolute(arr_Psi)**2)*dt)*pre_expV
    else:
        arr_Psi = np.real(arr_Psi).astype("float")

    # -*- Bogoliubov-de Gennes equation -*-
    
    # Sub-diagonal part calL
    beta = np.diag(2*h**(-2) + V(x) - mu + 2*gN*arr_Psi**2)
    calL = beta - alpha
    
    # Diagonal part calM
    calM = np.diag(gN*arr_Psi**2)
    
    # BdG matrix
    T = np.c_[np.r_[calL, calM], np.r_[-calM, -calL]]

    # Calculate eigenvalue problem
    wr, wi, vl, vr, info = dgeev(T, compute_vl=0, compute_vr=0, overwrite_a=True)
    wi.sort()
    imag_igen.append(wi[-1])


# --*-- Plot --*--

plt.plot(arr_gN, imag_igen)
plt.xlim(arr_gN[0], arr_gN[-1])
plt.show()

