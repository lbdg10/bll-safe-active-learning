#%% Import libraries
from datetime import datetime
from typing import List, NamedTuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import ssnet

TrainTBTT = NamedTuple('TrainTBTT', Ts=int, Ns=int) # to create tuples

# %% Load data
def ensure2D(x: np.ndarray):
    if x.ndim == 1:
        return np.expand_dims(x, axis=1)
    return x

# Select input and output variables
U_train = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Train_Aroma_30ggPreal.csv', delimiter=',',  usecols = (0)))
Y_train = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Train_Aroma_30ggPreal.csv', delimiter=',', usecols = (8)))
U_val = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Val_Aroma_30ggPreal.csv', delimiter=',',  usecols = (0)))
Y_val = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Val_Aroma_30ggPreal.csv', delimiter=',',  usecols = (8)))
U_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Aroma_30ggPreal.csv', delimiter=',',  usecols = (0)))
Y_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Aroma_30ggPreal.csv', delimiter=',',  usecols = (8)))


# %% Model training

# Size of inputs and outputs
size_inp = 1
size_out = 1

# Ns: number of subsequences we want to extract for training/validation/testing
# Ts: length of each training/validation/testing subsequence 
def train_nnarx_model(ffnn_layers: List[int], horizon: int, deltaiss_regularizer: torch.nn.Module, train_batch_size: int, 
                    Ts: int = 200, Ns: int = 200, iss_regularizer: torch.nn.Module = None, 
                    dropout: float = -1.0, lr: float = 1e-3):
    # Scale data
    input_scaler = ssnet.data.MinMaxSequenceScaler()
    output_scaler = ssnet.data.MinMaxSequenceScaler()

    # Select the proper datasets
    dataset = ssnet.data.tbptt(training=(U_train, Y_train), validation=(U_val, Y_val), testing=(U_test, Y_test),
                               Ts_train=Ts, Ns_train=Ns, Ts_val=300, Ns_val=10,
                               input_scaler=input_scaler, output_scaler=output_scaler)

    train_loader = DataLoader(dataset.training, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation, batch_size=10, shuffle=False)
    test_loader = DataLoader(dataset.testing, batch_size=1, shuffle=False)
    
    # Select the features of the NN we want to train
    nnarx = ssnet.nn.StateSpaceNNARX(units=ffnn_layers, in_features=size_inp, out_features=size_out, horizon=horizon, 
                                     input_feedthrough=True)
    net = ssnet.nn.StateSpaceNN(nnarx, batch_first=True, input_scaler=input_scaler, output_scaler=output_scaler)
    net.init_optimizer(torch.optim.Adam, lr=lr)

    # Select callbacks and training parameters
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers_str = '-'.join([str(nu) for nu in ffnn_layers])
    descr_str = f'NNARX_{layers_str}_H{horizon}_bs{train_batch_size}_Ts{Ts}_Ns{Ns}_{current_time}'

    callbacks = ssnet.callbacks.CallbacksWrapper(tensorboard_instance=f'training_output/Aroma/{descr_str}', 
                                             matfile_instance=f'training_output/Aroma/{descr_str}/net.mat',
                                             callbacks=[ssnet.callbacks.SigIntCallback(), 
                                                        ssnet.callbacks.TargetMetricCallback(1e-5), 
                                                        ssnet.callbacks.EarlyStoppingCallback(patience=500, watch_from=100),
                                                        ssnet.callbacks.LoggingCallback(),
                                                        ssnet.callbacks.MatlabExportCallback(),
                                                        ssnet.callbacks.PerformanceTestingCallback(test_loader, plot_fequency=100)])

    training_results = net.fit(criterion=torch.nn.MSELoss(), 
                               train_loader=train_loader, 
                               val_loader=val_loader, 
                               val_metric=torch.nn.MSELoss(),
                               iss_regularizer=iss_regularizer if deltaiss_regularizer is None else None, 
                               deltaiss_regularizer=deltaiss_regularizer,
                               callbacks=callbacks,
                               washout=20,
                               epochs=300)

    return training_results, descr_str


# %% Call the training
diss = ssnet.nn.SoftPieceWiseRegularizer(clearance=0.04, omega_plus=1e-3, omega_minus=1e-9, steepness=10.0)
training_results, descr_str = train_nnarx_model(ffnn_layers=[5], horizon=1, deltaiss_regularizer=diss,
                                                train_batch_size=20, Ts=200, Ns=200, lr=3e-3)
print(training_results["FIT"])
