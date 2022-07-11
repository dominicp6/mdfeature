import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
x = np.arange(-2, 2.5, 1/1000)

def function(x):
    return np.sin(x-2) * x**2 - 0.4 * x**2 + 0.2 * x

y = [function(val) for val in x]

prob = np.exp(-np.array(y))
normalised_prob = prob / np.max(prob) - 3
plt.vlines(x=0.09, ymin=-3, ymax=1, linestyles='dashed')
plt.vlines(x=-1.5, ymin=-3, ymax=1, linestyles='dashed')
plt.vlines(x=-0.6, ymin=-3, ymax=1, linestyles='dashed')
plt.vlines(x=1, ymin=-3, ymax=1, linestyles='dashed')
plt.text(x=-1.8, y=-0.65, s=r"$\mathbf{x_1}$", fontsize=14)
plt.text(x=-0.5, y=-0.65, s=r"$\mathbf{x_2}$", fontsize=14)
plt.text(x=0.15, y=0.1, s=r"$\mathbf{x_{max}}$", fontsize=14)
plt.text(x=1.1, y=-1, s=r"$\mathbf{A}$", fontsize=14)
plt.text(x=-2.3, y=-2.75, s=r"$p(x)$", fontsize=16, c='r')
plt.text(x=-2.3, y=1.3, s=r"$U(x)$", fontsize=16)
plt.plot(x, y, c='k')
plt.plot(x,normalised_prob, c='r')
plt.gca().fill_between(x, normalised_prob, -3, color='r')
plt.axis('off')
plt.savefig('kramersRate.pdf')
plt.show()
