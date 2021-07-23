#!/C:/Users/89721
#coding: utf-8
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path as path
import os

# remove the points in the Features name
# Input: String => Output: String
def removePoint(s):
    newS1 = ""
    for i in s:
        if (i == '|') or (i == '.') or (i == '<'):
            continue
        newS1 = newS1 + i
    return newS1

def removeMagpie(s):
    if ("MagpieData " in s):
        l = len("MagpieData ")
        newS2 = s[l:]
        return newS2
    else:
        return s

# read data and save data into list
path = os.path.dirname(os.path.realpath(__file__))
file1 = path + '/Features of original tenary alloys.xlsx'
df1 = pd.read_excel(file1)
df1 = df1.iloc[:, 2:]
featuresName = df1.columns.tolist()

file2 = path + '/Features of balanced ternary alloys.xlsx'
df2 = pd.read_excel(file2)
df2 = df2.iloc[:, 2:]


for i in featuresName[:-1]:

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
    plt.legend()
    plt.ylabel('Probability')
    temp_i_1 = removeMagpie(i)
    plt.xlabel(temp_i_1)
    temp_i_2 = removePoint(temp_i_1)
    saving_path = path + "/feature space distribution/" + temp_i_2 + ".tiff"
    plt.savefig(saving_path)
    plt.clf()
    