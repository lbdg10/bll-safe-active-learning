#%% Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplt
import numpy as np

#%% Compute power 
# Compute day of the week, month and hour 
data = pd.read_csv('ssnet\Datasets\Aroma\Train_Aroma_10ggPhigh.csv') 

cp = 4186
T_s = data.iloc[:, 0]
T_r = data.iloc[:, 21]
q = data.iloc[:, 22]
P_tot = cp * q * (T_s - T_r)
Ts_last = data.iloc[:, 8]
P1 = data.iloc[:, 1]
P2 = data.iloc[:, 2]
P3 = data.iloc[:, 3]
P4 = data.iloc[:, 4]
P5 = data.iloc[:, 5]

new_data = pd.DataFrame({'Ts': T_s, 'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'Ts_last': Ts_last, 'P_tot': P_tot})

#new_data.to_csv('ssnet\Datasets\Aroma\Train_Ts5_Phigh.csv') 
