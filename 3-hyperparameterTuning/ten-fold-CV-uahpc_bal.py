# coding: utf-8
# authour: Yi Yao
# edited by Timothy Sullivan

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os

# ===========================================================================================================================================

#Helper functions:

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

# input: List(AUC) + List(RMSE) + hyperparameters (Dropout value, layer 1 neurons, layer 2 neurons) => output: Dataframe ready to excel write
# Saves the mean and standard deviation of AUC and RMSE for the Kfold validation sets
def save_auc_rmse_per_fold(AUC_list, RMSE_list, Dval, L1val, L2val):
  mean_auc  = np.mean(AUC_list,  axis=0)
  std_auc   = np.std(AUC_list,  axis=0)
  mean_rmse = np.mean(RMSE_list, axis=0)
  std_rmse  = np.std(RMSE_list,  axis=0)
  L_sum = [Dval, L1val, L2val, mean_auc, std_auc, mean_rmse, std_rmse]
  df = pd.DataFrame(L_sum).T
  df.columns = ['Dropout Val', 'Layer 1 Neurons', 'Layer 2 Neurons', 'AUC', 'St Dev', 'RMSE', 'St Dev']
  return df

# input: List(dataframes) + List(sheet names) + String(path to excel file) => output: excel file
# Writes given dataframes to their corresponding sheet names in the excel file at path
def write_to_excel(df_list, sheet_list, path):
    with pd.ExcelWriter(path) as writer:
        for i in range(len(df_list)):
            df_list[i].to_excel(writer, header=True, index=False, sheet_name=sheet_list[i])


# =============================================================================================================================================
# Main function: 

# Set paths and load data

data_path = os.path.dirname(os.path.realpath(__file__))

# path = data_path + "/features/Features of original tenary alloys.xlsx"
path = data_path + "/features/Features of balanced ternary alloys.xlsx"
X, Y, mean_norm, range_norm= get_input_output(path)


#Result Dataframes. Must remain outside of all loops
perFold_df = pd.DataFrame(columns = ['Dropout Val', 'Layer 1 Neurons', 'Layer 2 Neurons', 'AUC', 'St Dev', 'RMSE', 'St Dev'])

for dv in [0, 0.01, 0.05, 0.1, 0.2, 0.5]:
    for l_1 in [50, 100, 150, 200, 250]:
        for l_2 in [10, 15, 20, 25, 50]:

            # Initialization Hyperparameters
            dropout_val = dv
            layer_1_neurons = l_1
            layer_2_neurons = l_2
            layer_1_activation = 'sigmoid'
            layer_2_activation = 'sigmoid'
            output_layer_activation = 'sigmoid'
            num_epochs = 20 #We could consider the number of epochs a hyperparameter in this model. There might be a way to train it with callbacks.EarlyStopping()

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=10, shuffle=True)

            # some list to save data
            auc_per_fold  = []
            rmse_per_fold = []
            L_auc  = []  # save auc  for each single alloy system
            L_rmse = []  # save RMSE for each single alloy system

            # K-fold Cross Validation model evaluation
            fold_no = 1

            for train, test in kfold.split(X, Y):

              # Define the model architecture

              model = tf.keras.models.Sequential([
                tf.keras.Input(shape=(1,131)),
                tf.keras.layers.Dense(layer_1_neurons, activation=layer_1_activation),
                tf.keras.layers.Dropout(dropout_val),
                tf.keras.layers.Dense(layer_2_neurons, activation=layer_2_activation),
                tf.keras.layers.Dense(1, activation=output_layer_activation)])

              # Compile the model
              model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC','RootMeanSquaredError'])

              # Generate a print
              print('------------------------------------------------------------------------')
              print(f'Training for fold {fold_no} ...')

              # Fit data to model
              history = model.fit(x=X[train], y=Y[train],
                          batch_size=1,
                          epochs=num_epochs,
                          verbose=2)

              # Generate generalization metrics
              scores = model.evaluate(X[test], Y[test], verbose=0)
              print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}')
              auc_per_fold.append(scores[1])
              rmse_per_fold.append(scores[2])

              # Increase fold number
              fold_no = fold_no + 1

            # Save data
            df1 = save_auc_rmse_per_fold(auc_per_fold, rmse_per_fold, dropout_val, layer_1_neurons, layer_2_neurons)
            perFold_df = perFold_df.append(df1, ignore_index=True)


#Final excel write. Must go outside all loops
#save_path = data_path + "/results/AUC_RMSE_original.xlsx"
save_path = data_path + "/results/AUC_RMSE_balanced.xlsx"

write_to_excel([perFold_df], ["perFold"], save_path)

