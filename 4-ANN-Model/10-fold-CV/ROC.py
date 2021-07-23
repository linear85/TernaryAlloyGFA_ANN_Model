#!/C:/Users/89721
#coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import os

path = os.path.dirname(os.path.realpath(__file__))
file1 = path + '/ROC by original dataset.xlsx'
df = pd.read_excel(file1, index_col=None, header=None)
x1 = df[0].tolist()
y1 = df[1].tolist()

file2 = path + '/ROC by balanced dataset.xlsx'
df = pd.read_excel(file2, index_col=None, header=None)
x2 = df[0].tolist()
y2 = df[1].tolist()

plt.plot(x1, y1, 'r', linewidth=2.0, label='original-AUC')
plt.plot(x2, y2, 'b', linewidth=2.0, label='balanced-AUC')
plt.plot([0, 1], [0, 1], '--', linewidth=1.0, label='random-AUC')
plt.xlabel('False Positive Rate', fontsize=15)
# ax1.set_ylim(0,1)
plt.ylabel('True Positive Rate', fontsize=15)
plt.tick_params(labelsize=10, width=1.0)
plt.legend(frameon=True, prop={'size':12})
plt.show()