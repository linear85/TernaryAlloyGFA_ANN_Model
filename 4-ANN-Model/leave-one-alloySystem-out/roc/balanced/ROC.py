import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from os import listdir
from pathlib import Path

def getValue(df, row, column):
	L =  df.iloc[row, column]
	if ('nan' in L ):
		return None
	L = re.findall('[0-9.]+', L)
	L = [float(i) for i in L]
	return L

labels = ["AlNiTi", "CoFeZr", "CoTiZr", "CoVZr", "FeTiNb", "AlCuFe", "AlFeGd", "AlFeNi", "AlMgTi", "BFeN", "BFeNb",
          "BFeZr", "CoFeNb", "CoMnNb", "CrGePd", "CrMoNi", "FeHfTa"]

mypath = os.path.dirname(os.path.realpath(__file__))

onlyfiles = [f for f in listdir(mypath) if '.xlsx' in f]

for i in onlyfiles:
	file_path = mypath + '/' + i
	df = pd.read_excel(file_path, header=None)
	Path(mypath+'/'+i[0:-5]).mkdir(parents=True, exist_ok=True)
	saving_path = mypath+'/'+i[0:-5]
	flag = False
	for num in range(20):
		FPR = getValue(df, 0, num)
		TPR = getValue(df, 1, num)
		if (FPR == None or TPR == None):
			flag = True
			break
		plt.plot(FPR, TPR)
		plt.xlabel('False Positive Rate', fontsize=12)
		plt.ylabel("True Positive Rate", fontsize=12)
		plt.title(i[0:-5]+'-'+str(num), fontsize=15)
		plt.savefig(saving_path+'/'+i[0:-5]+str(num)+'.tiff')
		plt.clf()

	if (flag):
		print(i[0:-5])
		continue

# file_path_Ori = path + '/ROC by original dataset.xlsx'
# file_path_Bal = path + '/ROC by balanced dataset.xlsx'
# df_Ori = pd.read_excel(file_path_Ori, header=None)
# df_Bal = pd.read_excel(file_path_Bal, header=None)
# for i in range(len(labels)):
# 	FPR_Ori = getValue(df_Ori, 0, i)
# 	TPR_Ori = getValue(df_Ori, 1, i)
# 	FPR_Bal = getValue(df_Bal, 0, i)
# 	TPR_Bal = getValue(df_Bal, 1, i)
# 	if (FPR_Ori == None or TPR_Ori == None or FPR_Bal == None or TPR_Bal == None):
# 		continue
# 	plt.plot(FPR_Ori, TPR_Ori, label = 'Original')
# 	plt.plot(FPR_Bal, TPR_Bal, label = 'Balanced')
# 	plt.legend(fontsize=12)
# 	plt.xlabel('False Positive Rate', fontsize=12)
# 	plt.ylabel("True Positive Rate", fontsize=12)
# 	plt.title(labels[i], fontsize=15)
# 	plt.show()
	