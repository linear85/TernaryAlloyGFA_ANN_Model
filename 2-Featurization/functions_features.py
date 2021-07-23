# functions to calculate VEC, entropy, enthalpy and Bulk modules
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import append
import pandas as pd
import math
import numpy as np
import os


data_path = os.path.dirname(os.path.realpath(__file__)) + '\element data'

file = data_path + "\element data.xlsx"
df = pd.read_excel(file, index_col=[1])

file2 = data_path + "\Delta H.xlsx"
df2 = pd.read_excel(file2, index_col=[0])


# ListOfComposition => [minimum, maximum, range, mean, average deviation] of VEC
#                      [minimum, maximum, range, mean, average deviation] of BULK
# produce the features of VEC and BULK
def calPro (L):
    L1 = [L[i] for i in range(len(L)) if i%2 ==0]                                    # Elements
    L2 = (np.array([L[i] for i in range(len(L)) if i%2 !=0]))/100                    # Compositions
    L_VEC = np.array([df.at[L1[i], "VEC"] for i in range(len(L1))])                  # VEC for correspoding elements
    L_BULK = np.array([df.at[L1[i], "Bulk Modulus /Gpa"] for i in range(len(L1))])   # BULK for correspoding elements

    meanVEC = sum(L2*L_VEC)/len(L_VEC)
    ave_dev_VEC = (sum(abs((L2*L_VEC - meanVEC)))) / len(L_VEC)
    VEC = [max(L_VEC), min(L_VEC), max(L_VEC)-min(L_VEC), meanVEC, ave_dev_VEC]

    meanBULK = sum(L2*L_BULK)/len(L_BULK)
    ave_dev_BULK = (sum(abs((L2*L_BULK - meanBULK)))) / len(L_BULK)
    BULK = [max(L_BULK), min(L_BULK), max(L_BULK)-min(L_BULK), meanBULK, ave_dev_BULK]
    return [VEC,BULK]


# ListOfComposition => Entropy
# produce the entropy of the given alloy
def cal_Smix(L):
    L1 = (np.array([L[i] for i in range(len(L)) if i%2 !=0]))/100                    # Compositons
    if 0 in L1:
        print(L)
    L2 = np.log(L1)
    Smix = -8.3145 * sum(L1*L2)
    return Smix

# ListOfComposition => [minimum, maximum, range, mean, average deviation] of Hmix
# produce the features for Hmix
def cal_Hmix(L):
    if len(L) == 2:
        return [0 for i in range(5)]
    L1 = [L[i] for i in range(len(L)) if i%2 ==0]                                    # Elements
    L2 = (np.array([L[i] for i in range(len(L)) if i%2 !=0]))/100                    # Compositions
    L3 =[]                                                                           # Hmix of i and j
    L4 =[]                                                                           # ci*cj
    for i in range(len(L1)-1):
        for j in range(i+1, len(L1)):
            temp = df2[L1[i]][L1[j]]
            if pd.isnull(temp):
                L3.append(df2[L1[j]][L1[i]])
            else:
                L3.append(temp)
            L4.append(L2[i]*L2[j])
    L3 = np.array(L3)
    L4 = np.array(L4)
    meanHmix = sum(L3*L4) / (len(L3))
    ave_dev_Hmix = (sum(abs((L3*L4 - meanHmix)))) / len(L3)
    return [max(L3), min(L3), max(L3)-min(L3), meanHmix, ave_dev_Hmix]
