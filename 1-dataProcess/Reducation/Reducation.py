# coding: utf-8
# authour: Yi Yao
# reduce the alloy number to 200

import pandas as pd
from pandas import ExcelWriter
import os

# Input: Listof_df + String(path) Out: excel file 
# Saving the list of df into one excel file at different sheet
def save_xls(list_dfs,list_sheet, xls_path):
    with ExcelWriter(xls_path) as writer:
        for name, df in zip(list_sheet, list_dfs):
            df.to_excel(writer,name, header=False, index=False)
        writer.save()



path = os.path.dirname(os.path.realpath(__file__))

file = path + '/original high-throughput ternary alloys.xlsx'
xls = pd.ExcelFile(file)
L_sheet_name = xls.sheet_names[1:]
L_df = []
L_sum = []

for sheet in L_sheet_name:
    df = pd.read_excel(file, sheet_name=sheet, header=None, index_col=None)
    df = df.sort_values(by=[1, 3, 5])
    interval = len(df)/200
    L = [int(i*interval) for i in range(200)]
    df = df.iloc[L]
    L_df.append(df)
    L_sum.extend(df.values.tolist())

saving_path1 = path + '/reduced high-throughput ternary alloys.xlsx'
save_xls(L_df, L_sheet_name, saving_path1)

saving_path2 = path + '/total reduced high-throughput ternary alloys.xlsx'
pd.DataFrame(L_sum).to_excel(saving_path2, header=False, index=False)
    







