# coding: utf-8
# authour: Yi Yao

from numpy.lib.function_base import append
from pandas.core.frame import DataFrame
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import os

# ===========================================================================================================================================

# Helper functions:

def AUC_RMSE_different_part(model, file, mean_norm, range_norm, L):
    Performance = []
    df = file[file["alloy system"].isin(L)]
    L_alloy = df.to_numpy()
    L_x = np.delete(L_alloy, 0, 1)
    L_x = np.delete(L_x, 0, 1)
    L_y = L_alloy[:,-1]
    L_x = np.delete(L_x, -1, 1)
    L_x = np.asarray(L_x).astype(np.float32)
    L_x = feature_normalize(L_x, mean_norm, range_norm)
    L_y = np.asarray(L_y).astype(np.float32)
    scores = model.evaluate(L_x, L_y, verbose=0)
    Performance.append([scores[1], scores[2]])
    return Performance


# input: np.array(unnormalized feature) + np.array(mean) + np.array(range) => np.array(normlized feature)
# normalize the feature by the given parameters
def feature_normalize(Feature, mean_value, range_value):
  Feature = Feature - mean_value
  Feature = Feature / range_value
  return Feature

# input: string(path of the file) => output: np.array(input/features for tensorflow model) + np.array(output/lable for tensorflow model) + np.array(mean) + np.array(range)
# generating the input and output for tensorflow model and parameters for the latter normalization according to the given file
def get_input_output(path):
  df = pd.read_excel(path)
  L = df.to_numpy()
  L1 = np.delete(L, 0, 1)
  L1 = np.delete(L1, 0, 1)
  Y = L1[:,-1]
  L1 = np.delete(L1, -1, 1)
  X = np.asarray(L1).astype(np.float32) # Features
  # max_norm = np.max(X, axis=0)
  # min_norm = np.min(X, axis=0)
  # range_norm = (max_norm - min_norm) + (10 ** -8)
  range_norm = np.std(X, axis=0) + (10 ** -8)
  range_norm = range_norm.reshape((1, 131))
  mean_norm = np.mean(X, axis=0)
  mean_norm = mean_norm.reshape((1, 131))
  X = feature_normalize(X, mean_norm, range_norm)
  Y = np.asarray(Y).astype(np.float32)  # Labels
  return X, Y, mean_norm, range_norm


# =============================================================================================================================================

# Main function: 

Dataset = "Original"
# Dataset = "Balanced"

L_T = ['AlCuFe','AlFeGd','AlFeNi','AlMgTi','BFeN','BFeNb','BFeZr','CoFeNb','CoMnNb','CrGePd','CrMoNi','FeHfTa']
L_H = ['AlNiTi','CoFeZr','CoTiZr','CoVZr','FeTiNb']

# load data
path = os.path.dirname(os.path.realpath(__file__))

if (Dataset == 'Original'):
  path1 = path + "/Features/Features of original tenary alloys.xlsx"
if (Dataset == 'Balanced'):
  path1 = path + "/Features/Features of balanced ternary alloys.xlsx"
X, Y, mean_norm, range_norm= get_input_output(path1)
X, Y, mean_norm, range_norm= get_input_output(path1)

# load data of RMSE calculation for each single alloy system
path2 = path + "/Features/Features of RMSE ternary alloys.xlsx"
df2 = pd.read_excel(path2)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

# some list to save data
auc_per_fold  = []
rmse_per_fold = []
L_T_P = []
L_H_P = []

# K-fold Cross Validation model evaluation
fold_no = 1

for train, test in kfold.split(X, Y):

  # Define the model architecture
  model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,131)),
    tf.keras.layers.Dense(250, activation='sigmoid'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(25, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

  # Compile the model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC',"RootMeanSquaredError"])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(X[train], Y[train],
              batch_size=1,
              epochs=20,
              verbose=1)

  # Generate generalization metrics
  scores = model.evaluate(X[test], Y[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
  auc_per_fold.append(scores[1])
  rmse_per_fold.append(scores[2])

  Per_H = AUC_RMSE_different_part(model, df2, mean_norm, range_norm, L_H)
  Per_T = AUC_RMSE_different_part(model, df2, mean_norm, range_norm, L_T)
  L_H_P.extend(Per_H)
  L_T_P.extend(Per_T)

  # Increase fold number
  fold_no = fold_no + 1

print(auc_per_fold)
print(rmse_per_fold)

if (Dataset == 'Original'):
  saving_path1 =  path + "/10-fold-CV/Original_high-throughput.xlsx"
  saving_path2 = path + "/10-fold-CV/Original_traditional.xlsx"

if (Dataset == 'Balanced'):
  saving_path1 =  path + "/10-fold-CV/Balanced_high-throughput.xlsx"
  saving_path2 = path + "/10-fold-CV/Balanced_traditional.xlsx"

pd.DataFrame(L_H_P).to_excel(saving_path1, index=False, header=False)
pd.DataFrame(L_T_P).to_excel(saving_path2, index=False, header=False)




