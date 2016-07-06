# coding: utf-8

import numpy as np
from scipy.integrate import simps
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn


# --*-- Set functions --*--

# Set initial function
def Psi_0(x):
    return x*np.exp(-x**2/2)

# Set potential
def V(x):
    return 25*np.sin(x/5)**2


# --*-- Set constants and variables for initial condition --*--

# Volume, Maximum x-step
L, xN = 16.0, 1024

# Interaction constant, Chemical potential
gN, mu = 50, 10

# Maximum time, Time step 
tMax, tN = 2, 256

# x-step, Time step
h, dt = L/xN, tMax/tN

# Set x-space
x = np.linspace(-L/2, L/2, xN)

# Set time propagator of poptential term except non-linear term
pre_expV = np.exp((mu - V(x))*dt)
expK = np.exp(-(2*np.pi*fftfreq(xN, d=1/xN)/L)**2*dt)

# --*-- Time evolution by symplectic numerical solution --*--

# For-loop on each value of gN
for gN in [10, 30, 50, 70, 90]:
    # Set operators for time evolution for every gNs
    arr_Psi = Psi_0(x)
    expV = np.exp((-gN*Psi_0(x)**2)*dt)*pre_expV
    
    # Time evolution
    for i in range(tN):    
        # Multipling by time propagator of potential term
        arr_Psi *= expV
        
        # Fourier transformation
        arr_Psi = fft(arr_Psi)
        
        # Multipling time propagator of kinetic term
        arr_Psi *= expK
        
        # Inverse fourier transformation
        arr_Psi = ifft(arr_Psi)
    
        # Correction of chemical potential mu
        mu -= (simps(arr_Psi**2, x) - 1)/(2*dt)
        
        # Normalization of order parameter arr_Psi
        arr_Psi /= np.sqrt(simps(np.real(arr_Psi**2), x))
        
        # Reconfigure expV
        expV = np.exp((-gN*arr_Psi**2)*dt)*pre_expV

    # Plot arr_Psi for present gN
    plt.plot(x, np.real(arr_Psi**2), label="gN = {0}".format(gN))


# --*-- Matplotlib configuration --*--

plt.plot(x, V(x)/50, label="Potential\n(provisional)")
plt.ylim(0, 0.5)
plt.xlabel("q-space")
plt.ylabel("Order parameter")
plt.title("Static soliton of Gross-Pitaevskii equation")
plt.legend(loc = 'center right')
plt.show()

