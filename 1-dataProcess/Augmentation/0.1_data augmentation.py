# coding: utf-8
# author: Yi Yao
# augment the data of ternary alloys with the step of 0.1%

import os
import pandas as pd
from scipy.spatial import distance
import numpy as np
from pandas import ExcelWriter


# helper function

# Input: List [[30, 50, 20, "AM"], ...] + int => Output List
# produce a list of alloy compositions augmented by the giving list of composition and interval
def augmentation(List, interval):
    L = []
    for i in List:
        label = i[-1]
        Lx = np.linspace(i[0]-interval, i[0]+interval, 3)
        Ly = np.linspace(i[1]-interval, i[1]+interval, 3)
        Lz = np.linspace(i[2]-interval, i[2]+interval, 3)
        for x in Lx:
            for y in Ly:
                for z in Lz:
                    if x+y+z == 100 and x > 0 and y > 0 and z > 0:
                        L.append([x, y, z, label])
    return L

# Input: List [[50, 20, 30, "AM"], ....] => Output: List
# remove the alloys which distance is less than 2% and have different structure
def checkDistance(List, dist):
    L = []
    for i in List:
        flag = True
        for j in List:
            if i == j or i[-1] == j[-1]:    continue
            d = distance.euclidean(i[:-1], j[:-1])
            if d <= dist: flag = False
        if flag and i not in L:    L.append(i)
    return L

# Input: List of Composition => Output: List of Composition
# reduce the number in the giving List to 200
def reduceTo200(List):
    L = sorted(List, key = lambda x:(x[0], x[1], x[2]))
    interval = len(L)/200
    L2 = [L[int(i*interval)] for i in range(200)]
    return L2

# Input: Listof_df + String(path) Out: excel file 
# Saving the list of df into one excel file at different sheet
def save_xls(list_dfs,list_sheet, xls_path):
    with ExcelWriter(xls_path) as writer:
        for name, df in zip(list_sheet, list_dfs):
            df.to_excel(writer,name, header=False, index=False)
        writer.save()

# Inout: List of Composition [[50, 20, 30, "AM"], ....] => Output: List of Composition
# remove duplication
def removeDuplication(List):
    L = []
    for i in List:
        if i not in L:  L.append(i)
    return L

# main function
path = os.path.dirname(os.path.realpath(__file__))
file = path + '/original traditional ternary alloy category.xlsx'
xls = pd.ExcelFile(file)
L_sheet_name = xls.sheet_names
L_df = []
L_sum = []
for sheet in L_sheet_name:
    df = pd.read_excel(file, sheet_name=sheet, header=None, index_col=None)
    L = df.values.tolist()
    name = [L[0][0], L[0][2], L[0][4]]
    L_composition = [[i[1], i[3], i[5], i[-1]] for i in L]
    interval = 0.1
    dist = 2
    L1 = checkDistance(L_composition, dist)  # remove the alloys which distance is less than 2% and have different structure
    L2 = augmentation(L1, interval)          # augmentation
    if len(L2) > 200:
        L2 = reduceTo200(L2)
    L2 = removeDuplication(L2)               # remove duplication
    for i in L2:                             # insert element name
        i.insert(0, name[0])
        i.insert(2, name[1])
        i.insert(4, name[2])
    L_sum.extend(L2)
    df = pd.DataFrame(L2)
    L_df.append(df)

saving_path1 = path + '/augmented traditional ternary alloys.xlsx'
save_xls(L_df, L_sheet_name, saving_path1)

saving_path2 = path + '/total augmented traditional ternary alloys.xlsx'
pd.DataFrame(L_sum).to_excel(saving_path2, header=False, index=False)







    



