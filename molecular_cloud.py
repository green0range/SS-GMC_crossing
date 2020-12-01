import matplotlib.pyplot as plt
import numpy as np

def n(x,z):
    A = 0.6
    I_m = 1
    n_c = 1000
    return n_c*(1-A**2)/(np.cosh(z/I_m)+A*np.cos(x/I_m))**2

x = np.linspace(-100,100,1000)
z = 5
n = n(x,z)

plt.plot(x, n)
plt.ylabel("particle number n")
plt.xlabel("x")
plt.show()