# coding: utf-8
# authour: Yi Yao
# Featurization

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import AtomicPackingEfficiency
from matminer.featurizers.composition import Miedema
import os
import pandas as pd
import sys
sys.path.append(os.path.realpath(__file__))
import functions_features

withLabel = False

path = os.path.dirname(os.path.realpath(__file__))
data_path = path + '/data/balanced ternary alloys.xlsx'

df_original = pd.read_excel(data_path, header=None)

if (len(df_original.columns) == 7):
    withLabel = True
    Structure = df_original[6].tolist()
    Structure = [1 if i == 'AM' else 0 for i in Structure]
    df_original = df_original.iloc[:,0:-1]

df_temp1 = pd.DataFrame()
df_temp1["formula"] = ['' for i in range(len(df_original))]
for i in range(len(df_original.columns)):
    df_temp1["formula"] = df_temp1["formula"] + df_original[i].map(str)


if __name__ == '__main__':
    # matminer part
    ape_feat = AtomicPackingEfficiency()
    md_feat = Miedema()
    df_temp1 = StrToComposition().featurize_dataframe(df_temp1, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df_temp1 = ape_feat.featurize_dataframe(df_temp1, col_id="composition", ignore_errors=True, return_errors=True)
    df_temp1 = ep_feat.featurize_dataframe(df_temp1, col_id="composition",ignore_errors=True)  # input the "composition" column to the featurizer
    # df = md_feat.featurize_dataframe(df, col_id="composition",ignore_errors=True)
    df_temp1 = df_temp1.drop(columns=['AtomicPackingEfficiency Exceptions'])

    # my own function part
    L1 = df_original.values.tolist()
    L_features = []
    for i in L1:
        # remove the empty element in the list
        while (0 in i):
            p = i.index(0)
            p1 = p-1
            i.pop(p)
            i.pop(p1)
        L_VEC_BULK = functions_features.calPro(i)
        Smix = functions_features.cal_Smix(i)
        L_Hmix = functions_features.cal_Hmix(i)
        L_temp = []
        L_temp.extend(L_VEC_BULK[0])
        L_temp.extend(L_VEC_BULK[1])
        L_temp.append(Smix)
        L_temp.extend(L_Hmix)
        L_features.append(L_temp)

    features_name = ["MaxVEC", "MinVEC", "RangeVEC", "MeanVEC", "AveDevVEC", "MaxBULK", "MinBULK", "RangeBULK", "MeanBULK", "AveDevBULK",
                    "Smix", "MaxHmix", "MinHmix", "RangeHmix", "MeanHmix", "AveDevHmix"]
    df_temp2 = pd.DataFrame(L_features, columns=features_name)

    # combine two part and delete extra columns
    df_final = pd.concat([df_temp1, df_temp2], axis=1)
    L = df_final.columns.tolist()
    L1 = [i for i in L if ("mode" not in i)]
    df_final = df_final.loc[:, df_final.columns.isin(L1)]

    # create a column (alloy system) at the beginning
    L = df_final["formula"].tolist()
    L_alloySystem = []
    for i in L:
        L_temp = []
        for j in i:
            if j.isalpha(): L_temp.append(j)
        L_alloySystem.append("".join(L_temp))
    df_final.insert (0, "alloy system", L_alloySystem)
    df_final = df_final.drop(columns=['formula'])
    if withLabel:
        df_final['Structure'] = Structure

    # change path to save data
    df_final.to_excel("Please enter the path you want to save the calculated feature", index=False)


