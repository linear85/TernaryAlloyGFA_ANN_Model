# coding: utf-8
# authour: Yi Yao

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import os

# ===========================================================================================================================================

# Helper functions:

# input: model(tensorflow) + DF(pandas) + DF => output: dict(AUC and RMSE for each alloy system)
# calculating the RMSE for each single alloy system under 10-fold CV
def AUC_RMSE_each_single_alloy_system(model, file, mean_norm, range_norm):
  AUC, RMSE = [], []
  L = file["alloy system"].unique().tolist()  # List to save all the name of alloy system
  for i in L:
    df = file[file["alloy system"].isin([i])]
    L_alloy = df.to_numpy()
    L_x = np.delete(L_alloy, 0, 1)
    L_x = np.delete(L_x, 0, 1)
    L_y = L_alloy[:,-1]
    L_x = np.delete(L_x, -1, 1)
    L_x = np.asarray(L_x).astype(np.float32)
    L_x = feature_normalize(L_x, mean_norm, range_norm)
    L_y = np.asarray(L_y).astype(np.float32)
    scores = model.evaluate(L_x, L_y, verbose=0)
    AUC.append(scores[1])
    RMSE.append(scores[2])
  return AUC, RMSE, L

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

# input: List(AUC) + List(RMSE) + List(index of alloy system) + String(path to save file)=> output: excel file
# saving mean and standard deviation of AUC and RMSE for each single alloy system into an excel file according to the giving lists
def save_auc_rmse_each_single_alloy_system(L1, L2, L, path):
  L_auc  = np.array(L1)
  L_rmse = np.array(L2)
  mean_auc  = np.mean(L_auc,  axis=0)
  std_auc   = np.std(L_auc,  axis=0)
  mean_rmse = np.mean(L_rmse, axis=0)
  std_rmse  = np.std(L_rmse,  axis=0)
  L_sum = [L, mean_auc, std_auc, mean_rmse, std_rmse]
  pd.DataFrame(L_sum).T.to_excel(path, header=False, index=False)

# input: model(tensorflow) + String(path of file) => List(prediction GFA for all alloys)
# predicting GFA for all alloys under 10-fold CV 
def pred_GFA(model, df, mean_norm, range_norm):
  L = df.to_numpy()
  # print(L.shape)
  L = np.delete(L,0,1)
  L = np.delete(L,0,1)
  L = np.asarray(L).astype(np.float32)
  L = feature_normalize(L, mean_norm, range_norm)
  GFA = model(L)
  GFA = GFA.numpy()
  GFA = GFA.ravel()
  GFA = GFA.tolist()
  return GFA


# =============================================================================================================================================

# Main function: 


Dataset = "Original"
# Dataset = "Balanced"

path = os.path.dirname(os.path.realpath(__file__))

# load data
if (Dataset == 'Original'):
  path1 = path + "/Features/Features of original tenary alloys.xlsx"
if (Dataset == 'Balanced'):
  path1 = path + "/Features/Features of balanced ternary alloys.xlsx"
X, Y, mean_norm, range_norm= get_input_output(path1)

# load data of RMSE calculation for each single alloy system
path2 = path + "/Features/Features of RMSE ternary alloys.xlsx"
df2 = pd.read_excel(path2)

# load data of GFA prediction
path_gfa = path + "/Features/Features of ternary alloys to be predicted.xlsx"
df_gfa = pd.read_excel(path_gfa)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

# some list to save data
auc_per_fold  = []
rmse_per_fold = []
L_auc  = []  # save auc  for each single alloy system
L_rmse = []  # save RMSE for each single alloy system
L_GFA  = []  # save predicted GFA for all alloys


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

  # get ROC data
  y_pred_keras = model.predict(X[test]).ravel()
  fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y[test], y_pred_keras)

  # Calculate RMSE for each single alloy system
  auc, rmse, L_alloy_system = AUC_RMSE_each_single_alloy_system(model, df2, mean_norm, range_norm)
  L_auc.append(auc)
  L_rmse.append(rmse)

  # predicte GFA for all alloys
  gfa = pred_GFA(model, df_gfa, mean_norm, range_norm)
  L_GFA.append(gfa)

  # Increase fold number
  fold_no = fold_no + 1

# save data
if (Dataset == 'Original'):
  saving_path1 = path + "/10-fold-CV/AUC and RMSE by original dataset.xlsx"
  saving_path2 = path + "/10-fold-CV/Predicted GFA by original dataset.xlsx"
  saving_path3 = path + "/10-fold-CV/ROC by original dataset.xlsx"
  saving_path4 = path + "/10-fold-CV/Per fold RMSE by original dataset.xlsx"

if (Dataset == 'Balanced'):
  saving_path1 = path + "/10-fold-CV/AUC and RMSE by balanced dataset.xlsx"
  saving_path2 = path + "/10-fold-CV/Predicted GFA by balanced dataset.xlsx"
  saving_path3 = path + "/10-fold-CV/ROC by balanced dataset.xlsx"
  saving_path4 = path + "/10-fold-CV//Per fold RMSE by balanced dataset.xlsx"

save_auc_rmse_each_single_alloy_system(L_auc, L_rmse, L_alloy_system, saving_path1)

L_GFA = np.array(L_GFA)
mean_GFA = np.mean(L_GFA, axis=0)
std_GFA  = np.std(L_GFA,  axis=0)
L_GFA = [mean_GFA, std_GFA]
pd.DataFrame(L_GFA).T.to_excel(saving_path2, index=False, header=False)

pd.DataFrame([fpr_keras, tpr_keras]).T.to_excel(saving_path3, index=False, header=False)

pd.DataFrame([auc_per_fold, rmse_per_fold]).T.to_excel(saving_path4, index=False, header=False)
