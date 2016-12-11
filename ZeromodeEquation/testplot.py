import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn


def g(x, alpha):
    beta = 1 / (2 * np.sqrt(2) * alpha)
    gamma = -1 / (6 * np.sqrt(2) * alpha**3)
    return  np.sqrt(1 / (2 * np.pi * alpha**2)) * np.exp(-x**2 / (2 * alpha**2)) * (beta + 3 * gamma * x**2)
    

def h(x, alpha):
    return np.sqrt(1 / (2 * np.pi * alpha**2)) * np.exp(-x**2 / (2 * alpha**2)) / (2 * alpha**2) * x


def psi(x, alpha):
    beta = 1 / (2 * np.sqrt(2) * alpha)
    gamma = -1 / (6 * np.sqrt(2) * alpha**3)
    return (1 / (2 * np.pi * alpha**2))**0.25 * np.exp(-x**2 / (4 * alpha**2) + 1j * (beta * x + gamma * x**3))


x = np.linspace(-0.1, 0.1, 500)
alpha = 0.855 * (1e6)**(-1/3)
# plt.plot(x, g(x, alpha), label='real')
# plt.plot(x, h(x, alpha), label='imag')
plt.plot(x, np.real(psi(x, alpha)), label='psi for real')
plt.plot(x, np.imag(psi(x, alpha)), label='psi for imag')

plt.xlim(-0.1, 0.1)

# plt.ylim(-150, 150)
# print(simps(g(x, 0.03), x) / simps(h(x, 0.03), x))
# plt.plot(x, f_real(x))
# plt.plot(x, f_imag(x))
plt.legend(loc='best')
plt.show()
