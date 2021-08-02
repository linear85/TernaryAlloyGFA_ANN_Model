#/ coding: utf-8
#/ author: Yi Yao
#/ data: 05/14/2021

''' Leave one alloy system out '''

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_curve

# Helper functions:

# Input: DF(pandas) + String(name of an alloy system) => np.array(features) + np.array(label)
# removing the data of the givining alloy system from the dataset for training and convert it to features and labels
def removeAlloySystem(df, name):
    df1 = df[~df["alloy system"].isin([name])]
    df1 = df1.to_numpy()
    L_Y = df1[:,-1]
    L_Y = np.asarray(L_Y).astype(np.float32)
    L_X = np.delete(df1, -1, 1)
    L_X = np.delete(L_X,  0, 1)
    L_X = np.delete(L_X,  0, 1)
    L_X = np.asarray(L_X).astype(np.float32)
    return L_X, L_Y

# Input: DF(pandas) + String(name of an alloy system) => "np.array(features)" or "np.array(features) + np.array(label)"
# only select the data of the givining alloy system from the dataset for testing or prediction
def getAlloySystem(df, name):
    df1 = df[df["alloy system"].isin([name])]
    df1 = df1.to_numpy()
    if df1.shape[1] == 133:  # distinct predData and testData, there is not label in predData.
        L_X = np.delete(df1,  0, 1)
        L_X = np.delete(L_X,  0, 1)
        L_X = np.asarray(L_X).astype(np.float32)
        return L_X
    L_Y = df1[:,-1]
    L_Y = np.asarray(L_Y).astype(np.float32)
    L_X = np.delete(df1, -1, 1)
    L_X = np.delete(L_X,  0, 1)
    L_X = np.delete(L_X,  0, 1)
    L_X = np.asarray(L_X).astype(np.float32)
    return L_X, L_Y

# input: np.array(unnormalized feature) + np.array(mean) + np.array(range) => np.array(normlized feature)
# normalize the feature by the given parameters
def feature_normalize(Feature, mean_value, range_value):
  Feature = Feature - mean_value
  Feature = Feature / range_value
  return Feature

# input: np.array(training feature) => output: np.array(mean) + np.array(range)
# getting the parameters for latter normalization from given features
def getNormPara(Lx):
    meanNorm = np.mean(Lx, axis=0)
    # maxNorm  = np.max(Lx,  axis=0)
    # minNorm  = np.min(Lx,  axis=0)
    # rangeNorm = maxNorm - minNorm + (10 ** -8)
    rangeNorm = np.std(Lx, axis=0) + (10 ** -8)
    return meanNorm, rangeNorm

# input: None => output: model(tensorflow)
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,131)),
        tf.keras.layers.Dense(250, activation='sigmoid'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(25, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC',"RootMeanSquaredError", "Accuracy"])
    return model

# input: List of elements => output: List of average and standard deviation of the elements
def get_AveAndStd(L):
    L = np.array(L)
    L_mean = np.mean(L, axis=0)
    L_std  = np.std(L, axis=0)
    return [L_mean, L_std]

#=======================================================================================================================================================

# Main function:

# Dataset = "Original"
Dataset = "Balanced"

L_label = ['AlNiTi','CoFeZr','CoTiZr','CoVZr','FeTiNb','AlCuFe','AlFeGd','AlFeNi','AlMgTi','BFeN','BFeNb',
           'BFeZr','CoFeNb','CoMnNb','CrGePd','CrMoNi','FeHfTa']

data_path = os.path.dirname(os.path.realpath(__file__))

# load data
if (Dataset == "Original"):
    path1 = data_path + "/Features/Features of original tenary alloys.xlsx"
if (Dataset == 'Balanced'):
    path1 = data_path + "/Features/Features of balanced ternary alloys.xlsx"
trainData = pd.read_excel(path1)

# load data of AUC and RMSE calculation for each single alloy system
path2 = data_path + "/Features/Features of RMSE ternary alloys.xlsx"
testData = pd.read_excel(path2)

# load data of GFA prediction
path_gfa = data_path + "/Features/Features of ternary alloys to be predicted.xlsx"
predData = pd.read_excel(path_gfa)

# some list to save data
L_auc  = []
L_rmse = []
L_pred_GFA = []

# leave-one-alloy-system-out
for i in L_label:

    print('------------------------------------------------------------------------')
    print(f'Training for fold {i} ...')
    # filter data for the single alloy system
    trainFeatureX, trainLabelY = removeAlloySystem(trainData, i)
    testFeatureX,  testLabelY  = getAlloySystem(testData, i)
    predFeatureX = getAlloySystem(predData, i)

    L_AUC_Temp = []
    L_RMSE_Temp = []
    L_GFA_Temp = []
    L_ROC_Temp = []

    # Normalization
    meanNorm, rangeNorm = getNormPara(trainFeatureX)
    trainFeatureX = feature_normalize(trainFeatureX, meanNorm, rangeNorm)
    testFeatureX  = feature_normalize(testFeatureX,  meanNorm, rangeNorm)
    predFeatureX  = feature_normalize(predFeatureX,  meanNorm, rangeNorm)


    # run the training, validation and prediction process 10 times to get a stable results
    for _ in range(20):

        model_keras = build_model()
        # training
        model_keras.fit(trainFeatureX, trainLabelY, batch_size=1, epochs=20, verbose=1)
        # validation
        scores = model_keras.evaluate(testFeatureX, testLabelY, verbose=0)
        L_AUC_Temp.append(scores[1])
        L_RMSE_Temp.append(scores[2])

        # prediction
        predGFA = model_keras(predFeatureX)
        predGFA = predGFA.numpy()
        predGFA = predGFA.ravel()
        L_GFA_Temp.append(predGFA)

        # get ROC data
        y_pred_keras = model_keras.predict(testFeatureX).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(testLabelY, y_pred_keras)
        L_ROC_Temp.append([fpr_keras, tpr_keras])
    
    # store data into list
    if (Dataset == "Original"):
        saving_path_roc = data_path + "/leave-one-alloySystem-out/roc/original/" + i + '.xlsx'
    if (Dataset == "Balanced"):
        saving_path_roc = data_path + "/leave-one-alloySystem-out/roc/balanced/" + i + '.xlsx'
    pd.DataFrame(L_ROC_Temp).T.to_excel(saving_path_roc, header=False, index=False)
    L_auc.append(get_AveAndStd(L_AUC_Temp))
    L_rmse.append(get_AveAndStd(L_RMSE_Temp))
    L_pred_GFA.extend(get_AveAndStd(L_GFA_Temp))

L_data_saving = [L_label, L_auc, L_rmse]

# save data into excel files
if (Dataset == "Original"):
    saving_path1 = data_path + "/leave-one-alloySystem-out/AUC and RMSE by original dataset.xlsx"
    saving_path2 = data_path + "/leave-one-alloySystem-out/predicted GFA by original dataset.xlsx"

if (Dataset == 'Balanced'):
    saving_path1 = data_path + "/leave-one-alloySystem-out/AUC and RMSE by balanced dataset.xlsx"
    saving_path2 = data_path + "/leave-one-alloySystem-out/predicted GFA by balanced dataset.xlsx"

pd.DataFrame(L_data_saving).T.to_excel(saving_path1, header=False, index=False)
pd.DataFrame(L_pred_GFA).T.to_excel(saving_path2, header=False, index=False)

