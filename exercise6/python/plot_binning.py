import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'log_binning'


def log(x):
    if x == 0:
        return 0
    else:
        return np.log10(x)


data = pd.read_csv(f'{FILENAME}.csv')
plt.plot(data['k'].apply(log), data['P_k'].apply(log))
plt.xlabel('log(k)')
plt.ylabel('log(P(k))')
plt.savefig(f'{FILENAME}.jpg')
plt.show()

