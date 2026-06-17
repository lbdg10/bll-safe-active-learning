#%% Import libraries
import scipy.io
import numpy as np
from pylab import *
import matplotlib.pyplot as pyplt
from sklearn.metrics import mean_squared_error
import torch

# Read .mat file containing the RNN weights and biases, extract parameters and scalers
file = scipy.io.loadmat('net_GRU.mat')
layers = file["layers"][0]

# NNARX
# U1 = layers[0][0]["weights"][0][0]["U.0"][0]
# W1 = layers[0][0]["weights"][0][0]["W.0"][0]
# b1 = layers[0][0]["weights"][0][0]["b.0"][0]
# U0 = layers[0][0]["weights"][0][0]["U0"][0]
# b0 = layers[0][0]["weights"][0][0]["b0"][0]

# GRU
n_x = 5
Wz = layers[0][0]["weights"][0][0]["Wzf"][0][:, :n_x].T
Uz = layers[0][0]["weights"][0][0]["Uzf"][0][:, :n_x].T
bz = layers[0][0]["weights"][0][0]["bzf"][0][:, :n_x].T
Wf = layers[0][0]["weights"][0][0]["Wzf"][0][:, n_x:].T
Uf = layers[0][0]["weights"][0][0]["Uzf"][0][:, n_x:].T
bf = layers[0][0]["weights"][0][0]["bzf"][0][:, n_x:].T
Wr = layers[0][0]["weights"][0][0]["Wr"][0].T
Ur = layers[0][0]["weights"][0][0]["Ur"][0].T
br = layers[0][0]["weights"][0][0]["br"][0].T
Uo = layers[1][0]["weights"][0][0]["weight"][0]
bo = layers[1][0]["weights"][0][0]["bias"][0].T

input_scaler_scale = file["input_scaler"][0]["scale"][0]
input_scaler_bias = file["input_scaler"][0]["bias"][0]
output_scaler_scale = file["output_scaler"][0]["scale"][0]
output_scaler_bias = file["output_scaler"][0]["bias"][0]

# Function needed to correctly read csv files
def ensure2D(x: np.ndarray):
    if x.ndim == 1:
        return np.expand_dims(x, axis=1)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# # Model prediction function
def get_states_RNN(x0, U):
    Y = []
    X = []
    xk = x0
    for k in range(U.shape[0]):
        uk = np.atleast_1d(U[k]).reshape(1, 1) 
        zk = sigmoid(Wz@uk + Uz@xk + bz)
        fk = sigmoid(Wf@uk + Uf@xk + bf)
        phi = np.tanh(Wr@uk + Ur@(fk*xk) + br)
        xk = zk * xk + (1-zk) * phi
        yk = Uo @ xk + bo
        X.append(xk)
        Y.append(yk)
    return np.array(Y), np.array(X)


# Function to recursively update Lambda_inv and Q, as in (13)-(14) Pavone
def recursive_update(phi_t, y_t, Lambda_inv_prec, Q_prec):
    phi_t = phi_t.view(-1, 1)  
    y_t = y_t.view(-1, 1)      
    # Lambda_inv update
    denom = 1.0 + torch.matmul(phi_t.T, Lambda_inv_prec @ phi_t)
    Lambda_inv_new = Lambda_inv_prec - ((Lambda_inv_prec @ phi_t) @ (Lambda_inv_prec @ phi_t).T) / denom
    # Q update
    Q_new = phi_t @ y_t.T + Q_prec 
    return Lambda_inv_new, Q_new

# Function to compute y predicted
def predict(phi_t, theta_mean, Lambda_inv, sigma2):
    y_pred = theta_mean.T @ phi_t 
    std = torch.sqrt((1 + phi_t.T @ Lambda_inv @ phi_t)*sigma2)
    return y_pred, std


############################### BLL ###############################

# Input/Output
U_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Aroma_30ggPreal.csv', delimiter=',',  usecols = (0), skip_header=1))
Y_test = ensure2D(np.genfromtxt('ssnet\Datasets\Aroma\Test_Aroma_30ggPreal.csv', delimiter=',',  usecols = (8), skip_header=1))

# Constant input
#U_test[:,0] = U_test[840,0]

U_test = (U_test - input_scaler_bias)/input_scaler_scale
Y_test = (Y_test - output_scaler_bias)/output_scaler_scale

# Call RNN model
x0 = np.random.uniform(-1, 1, size=(n_x, 1))
y_pred_RNN, x_RNN = get_states_RNN(x0, U_test[:,0])

y_pred_RNN = y_pred_RNN.squeeze(axis=2)
x_RNN = x_RNN.squeeze(axis=2)
x_RNN = torch.from_numpy(x_RNN)
x_RNN = x_RNN.to(torch.float32)
Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.to(torch.float32)
y_pred_RNN = torch.from_numpy(y_pred_RNN)
y_pred_RNN = y_pred_RNN.to(torch.float32)

# Add Gaussian noise to RNN predictions
noise_std = 0.1 
y_pred_RNN_noisy = y_pred_RNN + torch.randn_like(y_pred_RNN)*noise_std

# Create dataset
y_pred_RNN_noisy = torch.from_numpy(np.matrix(y_pred_RNN_noisy))
y_pred_RNN_noisy = y_pred_RNN_noisy.to(torch.float32)
data_stream = list(zip(x_RNN, y_pred_RNN_noisy))

# Get dimension
x_dim = x_RNN.shape[1]      
y_dim = Y_test.shape[1]     
phi_dim = x_dim + 1        

# Prior
Lambda_inv = torch.eye(phi_dim)
Q = torch.zeros((phi_dim, y_dim))         

# Compute y predicted over the test set
y_pred_BLL = np.zeros((Y_test.shape[0], Y_test.shape[1])) 
y_std_BLL = np.zeros((Y_test.shape[0], Y_test.shape[1])) 
W_BLL = np.zeros((Y_test.shape[0], phi_dim)) 
W_std_BLL = np.zeros((Y_test.shape[0], phi_dim)) 
k = 0
for x_t, y_t in data_stream:  
    phi_t = torch.cat([x_t.flatten(), torch.tensor([1.0])])  # add bias
    y_t = y_t.flatten().float()
    Lambda_inv, Q = recursive_update(phi_t, y_t, Lambda_inv, Q)
    theta_mean = Lambda_inv @ Q
    y_pred, std = predict(phi_t, theta_mean, Lambda_inv, noise_std**2)
    y_pred_BLL[k,:] = y_pred.detach().cpu().numpy()
    y_std_BLL[k,:] = std.detach().cpu().numpy()
    W_BLL[k,:] = theta_mean[:,0].detach().cpu().numpy()
    W_std_BLL[k,:] = np.sqrt(noise_std**2*np.diag(Lambda_inv.numpy()))
    k = k+1

# Denormalize
y_BLL_max = y_pred_BLL[:, 0] + 1.96 * y_std_BLL[:, 0]
y_BLL_max = (y_BLL_max*output_scaler_scale) + output_scaler_bias
y_BLL_max = (np.array(y_BLL_max)).squeeze()

y_BLL_min = y_pred_BLL[:, 0] - 1.96 * y_std_BLL[:, 0]
y_BLL_min = (y_BLL_min*output_scaler_scale) + output_scaler_bias
y_BLL_min = (np.array(y_BLL_min)).squeeze()

y_pred_RNN = (y_pred_RNN*output_scaler_scale) + output_scaler_bias
y_pred_RNN = (np.array(y_pred_RNN)).squeeze()
y_pred_BLL = (y_pred_BLL*output_scaler_scale) + output_scaler_bias
U_test = (U_test*input_scaler_scale) + input_scaler_bias


############### Plot ###############

# Real parameters
file = scipy.io.loadmat('net.mat')
layers = file["layers"][0]
U0 = layers[0][0]["weights"][0][0]["U0"][0]
b0 = layers[0][0]["weights"][0][0]["b0"][0]

fig, ax = pyplt.subplots(3, 1)
time = np.arange(0, Y_test.shape[0])

ax[0].plot(time, U_test)
ax[0].set_ylabel('Input [°C]', fontsize=10)
ax[0].set_xlim((0, Y_test.shape[0]))

ax[1].plot(time, y_pred_RNN, label='RNN')
ax[1].plot(time, y_pred_BLL, label='BLL')
ax[1].fill_between(time, y_BLL_min, y_BLL_max, color='orange', alpha=0.3, label='95% CI')
ax[1].legend(fontsize=10)
ax[1].set_ylabel('Output [°C]', fontsize=10)
ax[1].set_xlim((0, Y_test.shape[0]))

ax[2].plot(time, y_std_BLL)
ax[2].set_ylabel('Sigma', fontsize=10)
ax[2].set_xlim((0, Y_test.shape[0]))
pyplt.show()


fig, ax = pyplt.subplots(6, 1)
time = np.arange(0, Y_test.shape[0])

ax[0].plot(time, U0[0]*ones((Y_test.shape[0],1)), label='RNN')
ax[0].plot(time, W_BLL[:,0], label='BLL')
ax[0].fill_between(time, W_BLL[:, 0] - 1.96 * W_std_BLL[:, 0], W_BLL[:, 0] + 1.96 * W_std_BLL[:, 0], color='orange', alpha=0.3, label='95% CI')
ax[0].legend(fontsize=10)
ax[0].set_ylabel('Weight', fontsize=10)
ax[0].set_xlim((0, Y_test.shape[0]))

ax[1].plot(time, U0[1]*ones((Y_test.shape[0],1)), label='RNN')
ax[1].plot(time, W_BLL[:,1], label='BLL')
ax[1].fill_between(time, W_BLL[:, 1] - 1.96 * W_std_BLL[:, 1], W_BLL[:, 1] + 1.96 * W_std_BLL[:, 1], color='orange', alpha=0.3, label='95% CI')
ax[1].legend(fontsize=10)
ax[1].set_ylabel('Weight', fontsize=10)
ax[1].set_xlim((0, Y_test.shape[0]))

ax[2].plot(time, U0[2]*ones((Y_test.shape[0],1)), label='RNN')
ax[2].plot(time, W_BLL[:,2], label='BLL')
ax[2].fill_between(time, W_BLL[:, 2] - 1.96 * W_std_BLL[:, 2], W_BLL[:, 2] + 1.96 * W_std_BLL[:, 2], color='orange', alpha=0.3, label='95% CI')
ax[2].legend(fontsize=10)
ax[2].set_ylabel('Weight', fontsize=10)
ax[2].set_xlim((0, Y_test.shape[0]))

ax[3].plot(time, U0[3]*ones((Y_test.shape[0],1)), label='RNN')
ax[3].plot(time, W_BLL[:,3], label='BLL')
ax[3].fill_between(time, W_BLL[:, 3] - 1.96 * W_std_BLL[:, 3], W_BLL[:, 3] + 1.96 * W_std_BLL[:, 3], color='orange', alpha=0.3, label='95% CI')
ax[3].legend(fontsize=10)
ax[3].set_ylabel('Weight', fontsize=10)
ax[3].set_xlim((0, Y_test.shape[0]))

ax[4].plot(time, U0[4]*ones((Y_test.shape[0],1)), label='RNN')
ax[4].plot(time, W_BLL[:,4], label='BLL')
ax[4].fill_between(time, W_BLL[:, 4] - 1.96 * W_std_BLL[:, 4], W_BLL[:, 4] + 1.96 * W_std_BLL[:, 4], color='orange', alpha=0.3, label='95% CI')
ax[4].legend(fontsize=10)
ax[4].set_ylabel('Weight', fontsize=10)
ax[4].set_xlim((0, Y_test.shape[0]))

ax[5].plot(time, b0*ones((Y_test.shape[0],1)), label='RNN')
ax[5].plot(time, W_BLL[:,5], label='BLL')
ax[5].fill_between(time, W_BLL[:, 5] - 1.96 * W_std_BLL[:, 5], W_BLL[:, 5] + 1.96 * W_std_BLL[:, 5], color='orange', alpha=0.3, label='95% CI')
ax[5].legend(fontsize=10)
ax[5].set_ylabel('Bias', fontsize=10)
ax[5].set_xlim((0, Y_test.shape[0]))
pyplt.show()
