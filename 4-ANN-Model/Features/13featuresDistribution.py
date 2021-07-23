#!/C:/Users/89721
#coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

Features_name = [r"$VEC$", r"$σ_{VEC}$", r"$ΔS_{mix}$", r"$PE$", r"$σ_{PE}$", r"$R$", r"$δ$", r"$T_m$",
                 r"$σ_{T_m}$", r"$ΔH_{mix}$", r"$σ_{ΔHmix}$", r"$B$", r"$σ_{B}$"]

real_Feature = ['MeanVEC', 'AveDevVEC', 'Smix', 'MagpieData mean Electronegativity', 'MagpieData avg_dev Electronegativity',
                'MagpieData mean CovalentRadius', 'MagpieData avg_dev CovalentRadius', 'MagpieData mean MeltingT',
                'MagpieData avg_dev MeltingT', 'MeanHmix', 'AveDevHmix', 'MeanBULK', 'AveDevBULK', 'Structure']

# read data and save data into list
path = os.path.dirname(os.path.realpath(__file__))
file1 = path + '/Features of original tenary alloys.xlsx'
df1 = pd.read_excel(file1)
df1 = df1[real_Feature]
featuresName = df1.columns.tolist()

file2 = path + '/Features of balanced ternary alloys.xlsx'
df2 = pd.read_excel(file2)
df2 = df2[real_Feature]

print(df1)
print(df2)

num = 0
for i in real_Feature[:-1]:

    plt.subplot(3,5, num+1)
    
    df_temp1 = df1[[i, "Structure"]]
    df_temp2 = df2[[i, "Structure"]]

    S11 = (df_temp1.loc[df_temp1["Structure"] == 0])[i]
    S12 = (df_temp1.loc[df_temp1["Structure"] == 1])[i]
    n, bins = np.histogram(S11)
    n = n/sum(n)
    plt.plot(bins[:-1], n, linestyle='dotted', color="blue", label = 'original-CR')
    n, bins = np.histogram(S12)
    n = n/sum(n)
    plt.plot(bins[:-1], n, linestyle='solid', color="blue", label = 'original-AM')

    S21 = (df_temp2.loc[df_temp2["Structure"] == 0])[i]
    S22 = (df_temp2.loc[df_temp2["Structure"] == 1])[i]
    n, bins = np.histogram(S21)
    n = n/sum(n)
    plt.plot(bins[:-1], n, linestyle='dotted',color="red", label = 'balanced-CR')
    n, bins = np.histogram(S22)
    n = n/sum(n)
    plt.plot(bins[:-1], n, linestyle='solid',color="red", label = 'balanced-AM')

    plt.xlabel(Features_name[num], fontsize=12)
    if num == 12:
        plt.legend(frameon=False, prop={'size':8})
        plt.legend(bbox_to_anchor=(1.4, 0.5), loc='center left', fontsize=10) 
    if num == 0 or num == 5 or num == 10:
        plt.ylabel('Probability', fontsize=12)
    # plt.xticks([])
    plt.tick_params(labelsize=8)
    num += 1

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.3, hspace=0.4)
plt.show()