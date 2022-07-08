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
plt.plot(x,y, c='k')
plt.plot(x,normalised_prob, c='r')
plt.gca().fill_between(x, normalised_prob, -3, color='r')
plt.axis('off')
plt.show()
