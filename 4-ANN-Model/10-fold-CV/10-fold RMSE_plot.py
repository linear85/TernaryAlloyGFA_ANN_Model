import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))

labels = ["AlNiTi", "CoFeZr", "CoTiZr", "CoVZr", "FeTiNb", "AlCuFe", "AlFeGd", "AlFeNi", "AlMgTi", "BFeN", "BFeNb",
          "BFeZr", "CoFeNb", "CoMnNb", "CrGePd", "CrMoNi", "FeHfTa"]
x = np.arange(len(labels))

path1 = path + '/AUC and RMSE by original dataset.xlsx'
df = pd.read_excel(path1, index_col=None, header=None)
w1 = df[3].tolist()
path2 = path + '/AUC and RMSE by balanced dataset.xlsx'
df = pd.read_excel(path2, index_col=None, header=None)
w2 = df[3].tolist()

width = 0.35

plt.bar(x - width/2, w1, width, label='original')
plt.bar(x + width/2, w2, width, label='balanced')
plt.tick_params(labelsize=10, width=1.0)
plt.xticks(x, labels, rotation="45")
plt.xlabel("Ternary alloy system", fontsize=15)
plt.ylabel(r"$RMSE$", fontsize=15)
plt.legend(frameon=True, prop={'size':12})
plt.axvline(x=4.5,color='black')
plt.text(0, 0.3, "High-throughput", fontsize=12)
plt.text(11.0, 0.3, "Traditional", fontsize=12)

plt.subplots_adjust(bottom=0.2)

plt.show()  # to show the figure

