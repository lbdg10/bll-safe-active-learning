#%% Import libraries
from datetime import datetime
from typing import Callable, List, NamedTuple
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
U_train = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Train_Ts5_Phigh.csv', delimiter=',',  usecols = (1,2,3,4,5,6), skip_header=1))
Y_train = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Train_Ts5_Phigh.csv', delimiter=',', usecols = (7,8), skip_header=1))
U_val = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Val_Ts5_Phigh.csv', delimiter=',',  usecols = (1,2,3,4,5,6), skip_header=1))
Y_val = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Val_Ts5_Phigh.csv', delimiter=',',  usecols = (7,8), skip_header=1))
U_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Ts5_Phigh.csv', delimiter=',',  usecols = (1,2,3,4,5,6), skip_header=1))
Y_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Ts5_Phigh.csv', delimiter=',',  usecols = (7,8), skip_header=1))


# %% Model training

# Size of inputs and outputs
size_inp = 6
size_out = 2

# Ns: number of subsequences we want to extract for training/validation/testing
# Ts: length of each training/validation/testing subsequence 

def train_rnn_model(rnn_layers: List[int], rnn_class: Callable, deltaiss_regularizer: torch.nn.Module, train_batch_size: int, 
                    Ts: int = 200, Ns: int = 200, iss_regularizer: torch.nn.Module = None, 
                    dropout: float = -1.0, lr: float = 1e-3):
    input_scaler = ssnet.data.MinMaxSequenceScaler()
    output_scaler = ssnet.data.MinMaxSequenceScaler()

    dataset = ssnet.data.tbptt(training=(U_train, Y_train), validation=(U_val, Y_val), testing=(U_test, Y_test),
                               Ts_train=Ts, Ns_train=Ns, Ts_val=300, Ns_val=10,
                               input_scaler=input_scaler, output_scaler=output_scaler)

    train_loader = DataLoader(dataset.training, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation, batch_size=10, shuffle=False)
    test_loader = DataLoader(dataset.testing, batch_size=1, shuffle=False)

    layers = []
    size_out_vec = [2]
    for i, nu in enumerate(rnn_layers):            
        last_layer = i == len(rnn_layers) - 1
        input_size = size_inp if i == 0 else rnn_layers[i - 1]
        layers.append(rnn_class(in_features=input_size, units=nu, io_delay=last_layer))
        layers.append(torch.nn.Linear(in_features=nu, out_features=size_out_vec[i]))
        if dropout > 0.0 and dropout < 1.0:
            layers.append(torch.nn.Dropout(dropout))
    
    net = ssnet.nn.StateSpaceNN(layers=layers, batch_first=True, input_scaler=input_scaler, output_scaler=output_scaler)
    net.init_optimizer(torch.optim.Adam, lr=lr)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers_str = '-'.join([str(nu) for nu in rnn_layers])
    rnn_str = 'GRU' if rnn_class == ssnet.nn.StateSpaceGRU else 'LSTM'
    descr_str = f'{rnn_str}_{layers_str}_bs{train_batch_size}_Ts{Ts}_Ns{Ns}{"_iss" if iss_regularizer is not None else ""}'\
                f'{"_deltaiss" if deltaiss_regularizer is not None else ""}{"_dp" if dropout > 0.0 else ""}_{current_time}'

    callbacks = ssnet.callbacks.CallbacksWrapper(tensorboard_instance=f'training_output/Aroma/{descr_str}', 
                                             matfile_instance=f'training_output/Aroma/{descr_str}/net.mat',
                                             callbacks=[ssnet.callbacks.SigIntCallback(), 
                                                        ssnet.callbacks.TargetMetricCallback(1e-5), 
                                                        ssnet.callbacks.EarlyStoppingCallback(patience=2000),
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
                               washout=30,
                               epochs=200)

    return training_results, descr_str


# %% Train the model

# RNN model
diss = ssnet.nn.SoftPieceWiseRegularizer(clearance=0.04, omega_plus=1e-3, omega_minus=1e-9, steepness=10.0)
training_results, descr_str = train_rnn_model(rnn_layers=[6], rnn_class=ssnet.nn.StateSpaceGRU,
                                              Ts=200, Ns=200, deltaiss_regularizer=diss, train_batch_size=20, lr=3e-3)

print(training_results["FIT"])
