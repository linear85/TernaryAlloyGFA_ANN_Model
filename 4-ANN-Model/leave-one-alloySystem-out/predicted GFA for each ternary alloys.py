#!/C:/Users/89721
#coding: utf-8

# plotting predicted GFA for 17 ternary alloys

import matplotlib.pyplot as plt
import xlrd
import numpy as np
import pandas as pd
import ternary
import re
import os

L = []
for x in range(0, 101, 2):
    for y in range(0, 101, 2):
        for z in range(0, 101, 2):
            if x + y + z == 100:
                L.append((x,y,z))

print(len(L))

L_label = ['AlNiTi','CoFeZr','CoTiZr','CoVZr','FeTiNb','AlCuFe','AlFeGd','AlFeNi','AlMgTi','BFeN','BFeNb','BFeZr','CoFeNb','CoMnNb','CrGePd','CrMoNi','FeHfTa']
L_sublabel = [re.findall('[A-Z][^A-Z]*', i) for i in L_label]

data_path = os.path.dirname(os.path.realpath(__file__))

# file = data_path + '/predicted GFA by original dataset.xlsx'
file = data_path + '/predicted GFA by balanced dataset.xlsx'
column = [i for i in range(34) if i%2 == 0]
df = pd.read_excel(file, header=None, index_col=None, usecols=column)
GFA = [df[i].tolist() for i in range(0, 34, 2)]


file2 = data_path + '/ternary alloys system with AC.xls'
wb = xlrd.open_workbook(filename=file2)
Names = wb.sheet_names()
L_exp = []
for i in range(len(Names)):
    sheet = wb.sheet_by_index(i)
    L_temp = [sheet.row_values(i) for i in range(sheet.nrows)]
    L_temp = [(float(i[1]), float(i[3]), float(i[5]), i[6]) for i in L_temp]
    L_exp.append(L_temp)


for i in range(17):

    d = {}
    for j in range(len(L)):
        d[L[j]] = GFA[i][j]

    scale = 100
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="blue")
    fontsize = 12
    offset = 0.15
    tax.left_axis_label(L_sublabel[i][0], fontsize=fontsize, offset=offset)
    tax.right_axis_label(L_sublabel[i][1], fontsize=fontsize, offset=offset)
    tax.bottom_axis_label(L_sublabel[i][2], fontsize=fontsize, offset=offset)
    if L_label[i] in Names:
        n = Names.index(L_label[i])
        L_AM = [x for x in L_exp[n] if x[-1] == 'AM']
        L_CR = [x for x in L_exp[n] if x[-1] == 'CR']
        L_AC = []
        # L_AC = [x for x in L_exp[n] if x[-1] == 'AC']
        if L_CR != []:
            tax.scatter(L_CR, label="CR", marker='x', color='blue', s=40)
        if L_AM != []:
            tax.scatter(L_AM, label="AM", marker='x', color='red', s=40)
        if L_AC != []:
            tax.scatter(L_AC, label="AC", marker='x', color='green', s=40)
    # tax.scatter(S2[i], marker='+', color='red', label="AC", s=100, alpha = 1)
    # tax.scatter(L_AgCuFe1, marker='+', color='blue', label="CR", s=100, alpha = 1)
    # tax.scatter(L_AgCuFe2, marker='+', label="AM", s=100, alpha = 1)
    tax.heatmap(d, style="h", vmin=0, vmax=1) # style could be 't', 'h' or 'd'
    # tax.scatter(S2[i], marker='o', color='red', label="AC", s=50, alpha = 1)
    tax.legend()
    tax.boundary()
    tax.set_title(L_label[i], fontsize=20)
    tax.ticks(axis='lbr', linewidth=1, multiple=20, offset=0.025)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.savefig(data_path + '/predicted GFA/Predicted GFA for '+L_label[i]+'.tif')
    # tax.show()