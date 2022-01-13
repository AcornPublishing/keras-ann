import csv
import numpy as np
import matplotlib.pyplot as plt

u = np.r_[
    np.ones(10),
    np.zeros(10),
    np.ones(10),
    np.zeros(10),
    np.ones(10),
    np.zeros(10),
    np.ones(10),
    np.zeros(10),
    np.linspace(0, 1, 20),
    np.linspace(1, 0, 40),
    np.random.random(60),
]
x = np.zeros(len(u))

for i in range(2, len(u)):
    x[i] = (0.98 + 0.01 * np.tanh(0.1 * x[i - 2])) * x[i - 1] + (
        0.04 + 0.01 * np.tanh(u[i - 1])
    ) * u[i - 2]

plt.figure()
plt.plot(u, label="u(input)")
plt.plot(x, label="x(state)")
plt.xlim(0, len(u))
plt.ylim(0, 2)
plt.legend()
plt.show()

np.savetxt("nonlin_exp.csv", np.c_[x, u], delimiter=",", fmt="%.5f")
