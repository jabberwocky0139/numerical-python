import numpy as np
import matplotlib.pyplot as plt
import seaborn

def f_real(x):
    return -x**2 * np.sin(4 * x) * np.exp(-3 * x**2)

def f_imag(x):
    return np.sin(x + x**3) * np.exp(-x**2)

def g(x):
    return np.exp(-x**3)


x = np.linspace(-4, 4, 500)
plt.plot(x, g(x))
# plt.plot(x, f_real(x))
# plt.plot(x, f_imag(x))
plt.ylim(-1, 1)
plt.show()
