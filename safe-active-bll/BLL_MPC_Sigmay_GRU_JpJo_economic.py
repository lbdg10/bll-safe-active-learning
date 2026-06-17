#%% Import libraries
import casadi as ca
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as pyplt
import random
import time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.io import savemat

# Seed everything
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################ Set values ################################################

# Parameters
N = 288
n_u = 1
n_d = 5
n_y = 2
n_x = 6
n_theta = (n_x + 1)*n_y
H = 24
sigma2 = 0.001
coeff = 5
epsilon_sigma = np.sqrt(sigma2)*coeff
delta_prob = 0.01
n_start = 1

# Read .mat file containing the GRU weights and biases, extract parameters and scalers
file = scipy.io.loadmat('net_T_Phigh.mat')
layers = file["layers"][0]
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

# Constant diturbances
d_const = np.array([-225723, -279507, -218707, -288861, -221046]).reshape(n_d, 1)
d_const = (d_const - input_scaler_bias[0][1:].reshape(-1, 1)) / input_scaler_scale[0][1:].reshape(-1, 1)

# Minimum and maximum bounds
y_min_T_variable = np.full(N+H, 70)
y_min_T_variable[:84] = 60
y_min_T_variable[252:] = 60 
y_max_T_value = 100 
y_min_P_value = np.ones(N+H)*0
y_max_P_value = 2700000
y_min_variable = np.vstack([y_min_T_variable, y_min_P_value])
y_max = torch.tensor([y_max_T_value, y_max_P_value])
y_min_variable = (y_min_variable - np.asarray(output_scaler_bias).reshape(n_y, 1)) / np.asarray(output_scaler_scale).reshape(n_y, 1)
y_max = (y_max - output_scaler_bias) / output_scaler_scale
y_min_variable_ca = ca.DM(y_min_variable)
y_max_ca = ca.DM(y_max.detach().cpu().numpy()).reshape((n_y, 1))
u_min = (70 - input_scaler_bias[0][0]) / input_scaler_scale[0][0]
u_max = (90 - input_scaler_bias[0][0]) / input_scaler_scale[0][0]
x_min = -1
x_max = 1
DeltaU_max = 1 / input_scaler_scale[0][0]

# Reference and price
Y_ref_den = np.zeros((N, n_y))
Y_ref_den[:,0] = 80
Y_ref_den[:,1] = 1243487
Y_ref_vec = (Y_ref_den - output_scaler_bias.reshape(1, n_y)) / output_scaler_scale.reshape(1, n_y)
price = np.ones(N)
price[0:72] = 1
price[72:144] = 4
price[144:216] = 1
price[216:288] = 4

# Weights of the cost function and Lipsitchz
w_Tdiff = 1.6
w_U = 2
w_slack = 0.001
w_J = 0.0001
Lips = np.maximum(np.max(price), w_Tdiff)
switching_threshold = 2*epsilon_sigma*H*Lips

# Function to compute y predicted by BLL: different mean but same variance
def predict(phi, theta_mean, Lambda_inv):
    phi = phi.view(-1)
    y_pred = torch.zeros(n_y)
    for i in range(n_y):
        index = slice(i*(n_x+1), (i+1)*(n_x+1))
        theta_i = theta_mean[index].view(-1) # select the correct parameters for that output
        y_pred[i] = theta_i @ phi
    index = slice(0, n_x+1)
    y_var = sigma2 * (phi @ Lambda_inv[index, index] @ phi)
    y_std = torch.sqrt(y_var)
    return y_pred, y_std

# MPC initialization
x0 = np.random.uniform(-1, 1, size=n_x)
u0_scaled = (80 - input_scaler_bias[0][0]) / input_scaler_scale[0][0] 
U_prev = np.ones((n_u, H)) * u0_scaled
X_prev = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_pess = np.ones((n_u, H)) * u0_scaled
X_prev_pess = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_opt = np.ones((n_u, H)) * u0_scaled
X_prev_opt = 2 * np.random.rand(n_x, H + 1) - 1
U_applied = []
y_RNN = []
switched_logic = np.ones((N, 1))
switched = False
n = 0
slack_tol = 1e-6
t_pess = np.zeros(N)
t_opt = np.zeros(N)
t_expl = np.zeros(N)
t_total = np.zeros(N)

# BNN prior
theta_star = np.concatenate((Uo.flatten(), bo.flatten()), axis=0).reshape(-1, 1) # actual parameters
theta_0 = theta_star*0.3 # initial guess of the mean
lambda_param = 0.3
Lambda_0 = np.eye(n_theta)*lambda_param # initial guess of the variance
diff = theta_star - theta_0
C = float(diff.T @ Lambda_0 @ diff)
Lambda_inv = torch.eye(n_theta)*1/lambda_param
theta_mean = torch.from_numpy(theta_0)
Q = torch.inverse(Lambda_inv)@theta_mean
theta_std = torch.sqrt(sigma2*torch.diag(Lambda_inv))
x0_BNN = (torch.from_numpy(np.array(x0))).to(torch.float32)
y_pred, y_std = predict(torch.cat([x0_BNN.squeeze(), torch.tensor([1.0])]), theta_mean, Lambda_inv)
y_BLL = np.zeros((N, n_y)) 
y_std_BLL = np.zeros((N, 1)) 
W_BLL = np.zeros((N, n_theta)) 
W_std_BLL = np.zeros((N, n_theta)) 
J_pess = np.zeros((N, 1)) 
J_opt = np.zeros((N, 1)) 
slack_vector = np.zeros((1, H))
max_slack_matrix = np.zeros((N, 1))
beta_n = np.sqrt(2 * np.log(1 / delta_prob)) + np.sqrt(C / sigma2)

Lambda_inv_store = torch.zeros((N, n_theta, n_theta), dtype=Lambda_inv.dtype, device=Lambda_inv.device)


################################################ Functions ################################################

# Sigmoid for GRU
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# One step GRU for MPC (Casadi)
def RNN_model_1step_MPC(xk, uk, dk):
    uk = ca.reshape(uk, (n_u, 1)) 
    dk = ca.reshape(dk, (n_d, 1))
    uk_tot = ca.vertcat(uk, dk)
    xk = ca.reshape(xk, (n_x, 1))
    zk = sigmoid(Wz@uk_tot + Uz@xk + bz)
    fk = sigmoid(Wf@uk_tot + Uf@xk + bf)
    phi = ca.tanh(Wr@uk_tot + Ur@(fk*xk) + br)
    xkp = zk * xk + (1-zk) * phi     
    ykp = Uo @ xkp + bo 
    return ykp, xkp

# One step GRU
def RNN_model_1step(xk, uk, dk):
    uk = np.atleast_1d(uk).reshape(n_u, 1) 
    dk = np.atleast_1d(dk).reshape(n_d, 1)
    uk_tot = np.vstack((uk, dk))
    xk = xk.reshape(n_x, 1) 
    zk = sigmoid(Wz@uk_tot + Uz@xk + bz)
    fk = sigmoid(Wf@uk_tot + Uf@xk + bf)
    phi = np.tanh(Wr@uk_tot + Ur@(fk*xk) + br)
    xkp = zk * xk + (1-zk) * phi
    ykp = Uo @ xkp + bo 
    return ykp, xkp

# Function to recursively update Lambda_inv and Q, as in (13)-(14) Pavone
def recursive_update(phi, y, Lambda_inv_prev, Q_prev):
    Q_new = Q_prev.clone()
    Lambda_inv_new = Lambda_inv_prev.clone()
    for i in range(n_y):
        index = slice(i*(n_x+1), (i+1)*(n_x+1))
        phi_i = phi.view(-1,1)
        y_i = y[i].view(1,1)
        denom = 1 + phi_i.T @ Lambda_inv_prev[index,index] @ phi_i
        Lambda_inv_new[index,index] -= (Lambda_inv_prev[index,index] @ phi_i @ phi_i.T @ Lambda_inv_prev[index,index]) / denom
        Q_new[index] += phi_i * y_i
    return Lambda_inv_new, Q_new

# Function to get the first h where an informative state occurs
def get_informative_horizon(slack_seq, tol):
    slack_seq = np.atleast_1d(slack_seq).flatten()
    for h in range(len(slack_seq)):
        if abs(slack_seq[h]) <= tol:
            return h
    return len(slack_seq) - 1


# MPC for pessimistic exploration
def mpc_exploration(x_n, Lambda_inv_n, theta_mean_n, theta_std_n, y_pred_n, y_std_n, U_prev, X_prev, n):

    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    lower_bound = opti.variable(n_y, H)
    upper_bound = opti.variable(n_y, H)
    Sigma_y = opti.variable(1, H)
    slack_sigma = opti.variable(1, H)
    epigrafica = opti.variable(1, 1)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    
    # Slack variables non negativity
    opti.subject_to(ca.vec(slack_sigma) >= 0)

    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    if n+H <= N:
        Y_ref = Y_ref_vec[n:n+H,:].T
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.vstack([Y_ref_vec[n:, :], np.tile(Y_ref_vec[-1, :], (dim_left, 1))]).T
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    Y_ref_ca = ca.DM(Y_ref)
    current_price_ca = ca.DM(current_price)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # Maximum difference between two consecutive inputs
        if h == 0:
            opti.subject_to(U[:,h] - U_prev[:,0] <= DeltaU_max)
            opti.subject_to(U[:,h] - U_prev[:,0] >= -DeltaU_max)
        else:
            opti.subject_to(U[:,h] - U[:,h-1] <= DeltaU_max)
            opti.subject_to(U[:,h] - U[:,h-1] >= -DeltaU_max)

        # RNN model
        y_pred_RNN, x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h], d_const)
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Uncertainty constraint
        phi_h = ca.vertcat(X[:,h], 1)
        index_start = 0
        index_end = n_x + 1
        opti.subject_to(Sigma_y[0, h] == ca.sqrt(sigma2 * ca.mtimes([phi_h.T, Lambda_inv_prev[index_start:index_end, index_start:index_end], phi_h])))
        opti.subject_to(beta_n*Sigma_y[0,h] >= epsilon_sigma - slack_sigma[0,h]) 

        # Predicted output
        y_hat_h = ca.vertcat(*[ca.mtimes(theta_mean_prev[i*(n_x+1):(i+1)*(n_x+1)].T, phi_h) for i in range(n_y)])
        
        # Pessimistic set constraint
        if n >= n_start:
            opti.subject_to(lower_bound[:,h] == y_hat_h - beta_n * Sigma_y[0,h])
            opti.subject_to(upper_bound[:,h] == y_hat_h + beta_n * Sigma_y[0,h])
            opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])
            opti.subject_to(lower_bound[:,h] >= y_min_variable_ca[:,n+h])
            opti.subject_to(upper_bound[:,h] <= y_max_ca)

        # Cost function: electrical price and slack
        if h==0:
            J_step += current_price_ca[h] * y_hat_h[1] + w_slack*slack_sigma[0, h] + w_U*(U[:,h] - U_prev[:,0])**2
        else:
            J_step += current_price_ca[h] * y_hat_h[1] + w_slack*slack_sigma[0, h] + w_U*(U[:,h] - U[:,h-1])**2

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])
    us = opti.variable(n_u, 1)
    xs = opti.variable(n_x, 1)
    eq_th = opti.variable(n_x, 1)
    opti.subject_to(us <= u_max)
    opti.subject_to(us >= u_min)
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    y_pred_RNN_s, x_pred_RNN_s = RNN_model_1step_MPC(xs, us, d_const)
    opti.subject_to(xs - eq_th <= x_pred_RNN_s)
    opti.subject_to(xs + eq_th >= x_pred_RNN_s)
    opti.subject_to(eq_th <= 0.01)
    opti.subject_to(eq_th >= 0)
    opti.subject_to(X[:,H]==xs)
    J_step += 1e2*eq_th.T@eq_th

    # Cost function: terminal cost
    opti.subject_to(y_hat_h[0]-Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(-y_hat_h[0]+Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(epigrafica >= 0)
    J_step += w_Tdiff*epigrafica

    # Minimize
    opti.minimize(w_J*J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)) )
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)) )
    opti.set_initial(slack_sigma, 0)
    opti.set_initial(Sigma_y, np.tile(y_std_n.item(), (1, H)))
    opti.set_initial(epigrafica, 0)

    # Solve problem
    opti.solver('ipopt', {
    "ipopt.max_iter": 3000,
    "ipopt.linear_solver": "mumps",  # deterministic linear solver
    "ipopt.print_level": 0})
    sol = opti.solve()
    U_opt = sol.value(U).reshape((n_u, H))
    slack_vector = sol.value(slack_sigma)
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)

    return U_opt, slack_vector, U_opt_tot, X_opt_tot


# Pessimistic MPC
def mpc_pessimistic(x_n, Lambda_inv_n, theta_mean_n, theta_std_n, y_pred_n, y_std_n, U_prev, X_prev, switched, n):
    
    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    lower_bound = opti.variable(n_y, H)
    upper_bound = opti.variable(n_y, H)
    Sigma_y = opti.variable(1, H)
    epigrafica = opti.variable(1, 1)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    
    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    if n+H <= N:
        Y_ref = Y_ref_vec[n:n+H,:].T
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.vstack([Y_ref_vec[n:, :], np.tile(Y_ref_vec[-1, :], (dim_left, 1))]).T
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    Y_ref_ca = ca.DM(Y_ref)
    current_price_ca = ca.DM(current_price)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # Maximum difference between two consecutive inputs
        if h == 0:
            opti.subject_to(U[:,h] - U_prev[:,0] <= DeltaU_max)
            opti.subject_to(U[:,h] - U_prev[:,0] >= -DeltaU_max)
        else:
            opti.subject_to(U[:,h] - U[:,h-1] <= DeltaU_max)
            opti.subject_to(U[:,h] - U[:,h-1] >= -DeltaU_max)

        # RNN model
        y_pred_RNN, x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h], d_const)
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Uncertainty constraint
        phi_h = ca.vertcat(X[:,h], 1)
        index_start = 0
        index_end = n_x + 1
        opti.subject_to(Sigma_y[0, h] == ca.sqrt(sigma2 * ca.mtimes([phi_h.T, Lambda_inv_prev[index_start:index_end, index_start:index_end], phi_h])))
        
        # Predicted output
        y_hat_h = ca.vertcat(*[ca.mtimes(theta_mean_prev[i*(n_x+1):(i+1)*(n_x+1)].T, phi_h) for i in range(n_y)])

        # Pessimistic set constraint
        if n >= n_start:
            opti.subject_to(lower_bound[:,h] == y_hat_h - beta_n * Sigma_y[0,h])
            opti.subject_to(upper_bound[:,h] == y_hat_h + beta_n * Sigma_y[0,h])
            opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])
            opti.subject_to(lower_bound[:,h] >= y_min_variable_ca[:,n+h])
            opti.subject_to(upper_bound[:,h] <= y_max_ca)

        # Cost function: electrical price
        if not switched:        
            J_step += current_price_ca[h] * y_hat_h[1] + ca.sum1(Lips*beta_n*Sigma_y[0,h])
        else:            
            if h==0:
                J_step += current_price_ca[h] * y_hat_h[1] + w_U*(U[:,h] - U_prev[:,0])**2
            else:
                J_step += current_price_ca[h] * y_hat_h[1] + w_U*(U[:,h] - U[:,h-1])**2

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])
    us = opti.variable(n_u, 1)
    xs = opti.variable(n_x, 1)
    eq_th = opti.variable(n_x, 1)
    opti.subject_to(us <= u_max)
    opti.subject_to(us >= u_min)
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    y_pred_RNN_s, x_pred_RNN_s = RNN_model_1step_MPC(xs, us, d_const)
    opti.subject_to(xs - eq_th <= x_pred_RNN_s)
    opti.subject_to(xs + eq_th >= x_pred_RNN_s)
    opti.subject_to(eq_th <= 0.01)
    opti.subject_to(eq_th >= 0)
    opti.subject_to(X[:,H]==xs)
    J_step += 1e2*eq_th.T@eq_th

    # Cost function: terminal cost
    opti.subject_to(y_hat_h[0]-Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(-y_hat_h[0]+Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(epigrafica >= 0)
    J_step += w_Tdiff*epigrafica

    # Minimize
    opti.minimize(w_J*J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(Sigma_y, np.tile(y_std_n.item(), (1, H)))
    opti.set_initial(epigrafica, 0)

    # Solve problem
    opti.solver('ipopt', {
    "ipopt.max_iter": 3000,
    "ipopt.linear_solver": "mumps",  # deterministic linear solver
    "ipopt.print_level": 0})
    sol = opti.solve()
    U_opt = sol.value(U[:,0])
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)

    return U_opt, sol.value(J_step), U_opt_tot, X_opt_tot


# Optimistic MPC
def mpc_optimistic(x_n, Lambda_inv_n, theta_mean_n, theta_std_n, y_pred_n, y_std_n, U_prev, X_prev, n):

    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    lower_bound = opti.variable(n_y, H)
    upper_bound = opti.variable(n_y, H)
    Theta = opti.variable(n_theta, H)
    Sigma_y = opti.variable(1, H)
    epigrafica = opti.variable(1, 1)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    
    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    if n+H <= N:
        Y_ref = Y_ref_vec[n:n+H,:].T
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.vstack([Y_ref_vec[n:, :], np.tile(Y_ref_vec[-1, :], (dim_left, 1))]).T
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    Y_ref_ca = ca.DM(Y_ref)
    current_price_ca = ca.DM(current_price)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # Maximum difference between two consecutive inputs
        if h == 0:
            opti.subject_to(U[:,h] - U_prev[:,0] <= DeltaU_max)
            opti.subject_to(U[:,h] - U_prev[:,0] >= -DeltaU_max)
        else:
            opti.subject_to(U[:,h] - U[:,h-1] <= DeltaU_max)
            opti.subject_to(U[:,h] - U[:,h-1] >= -DeltaU_max)

        # RNN model
        y_pred_RNN, x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h], d_const)
        opti.subject_to(X[:,h+1] == x_pred_RNN)
        
        # Uncertainty constraint
        phi_h = ca.vertcat(X[:,h], 1)
        index_start = 0
        index_end = n_x + 1
        opti.subject_to(Sigma_y[0, h] == ca.sqrt(sigma2 * ca.mtimes([phi_h.T, Lambda_inv_prev[index_start:index_end, index_start:index_end], phi_h])))
        
        # Predicted output
        y_hat_h = ca.vertcat(*[ca.mtimes(theta_mean_prev[i*(n_x+1):(i+1)*(n_x+1)].T, phi_h) for i in range(n_y)])
        y_theta_h = ca.vertcat(*[ca.mtimes(Theta[i*(n_x+1):(i+1)*(n_x+1), h].T, phi_h) for i in range(n_y)])

        # Optimistic set constraint
        opti.subject_to(lower_bound[:,h] == y_hat_h - beta_n * Sigma_y[0,h])
        opti.subject_to(upper_bound[:,h] == y_hat_h + beta_n * Sigma_y[0,h])
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])      
        
        if n >= n_start:
            if h == 0 or h == H-1: # invariant set
                opti.subject_to(lower_bound[:,h] >= y_min_variable_ca[:,n+h])
                opti.subject_to(upper_bound[:,h] <= y_max_ca)
            else: # optimistic set
                opti.subject_to(lower_bound[:,h] <= y_max_ca - 2*epsilon_sigma)
                opti.subject_to(upper_bound[:,h] >= y_min_variable_ca[:,n+h] + 2*epsilon_sigma) 
            
        # Constraint on y
        opti.subject_to(y_theta_h <= upper_bound[:,h])
        opti.subject_to(y_theta_h >= lower_bound[:,h])

        # Cost function: electrical price
        J_step += current_price_ca[h] * y_theta_h[1]

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])
    us = opti.variable(n_u, 1)
    xs = opti.variable(n_x, 1)
    eq_th = opti.variable(n_x, 1)
    opti.subject_to(us <= u_max)
    opti.subject_to(us >= u_min)
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
    y_pred_RNN_s, x_pred_RNN_s = RNN_model_1step_MPC(xs, us, d_const)
    opti.subject_to(xs - eq_th <= x_pred_RNN_s)
    opti.subject_to(xs + eq_th >= x_pred_RNN_s)
    opti.subject_to(eq_th <= 0.01)
    opti.subject_to(eq_th >= 0)
    opti.subject_to(X[:,H]==xs)
    J_step += 1e2*eq_th.T@eq_th

    # Cost function: terminal cost
    opti.subject_to(y_theta_h[0]-Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(-y_theta_h[0]+Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(epigrafica >= 0)
    J_step += w_Tdiff*epigrafica

    # Minimize
    opti.minimize(w_J*J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(Theta, np.tile(np.array(theta_mean_n).reshape(-1, 1), (1, H)))
    opti.set_initial(Sigma_y, np.tile(y_std_n.item(), (1, H)))
    opti.set_initial(epigrafica, 0)

    # Solve problem
    opti.solver('ipopt', {
    "ipopt.max_iter": 3000,
    "ipopt.linear_solver": "mumps",  # deterministic linear solver
    "ipopt.print_level": 0})
    sol = opti.solve()
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)

    return sol.value(J_step), U_opt_tot, X_opt_tot



################################################ Simulation ################################################

while n < N:
    # Pessimistic
    t0 = time.perf_counter()
    U_mpc_pess, J_pess[n,:], U_prev_pess, X_prev_pess = mpc_pessimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_pess, X_prev_pess, switched, n)
    t_pess[n] = time.perf_counter() - t0
    # Optimistic
    t0 = time.perf_counter()
    J_opt[n,:], U_prev_opt, X_prev_opt = mpc_optimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_opt, X_prev_opt, n)
    t_opt[n] = time.perf_counter() - t0

    U_prev_pess[:, :-1] = U_prev_pess[:, 1:] 
    U_prev_pess[:, -1] = U_prev_pess[:, -1]
    X_prev_pess[:, :-1] = X_prev_pess[:, 1:] 
    X_prev_pess[:, -1] = X_prev_pess[:, -1]
    U_prev_opt[:, :-1] = U_prev_opt[:, 1:] 
    U_prev_opt[:, -1] = U_prev_opt[:, -1]
    X_prev_opt[:, :-1] = X_prev_opt[:, 1:] 
    X_prev_opt[:, -1] = X_prev_opt[:, -1]

    # Switch logic
    if not switched and (J_pess[n,:] -J_opt[n,:] <= switching_threshold): 
        switched = True  

    if switched and (J_pess[n,:] -J_opt[n,:] > switching_threshold) :
        switched = False  

    if not switched:
        # Exploration
        t0 = time.perf_counter()
        U_mpc_exploration, slack_vector, U_prev, X_prev = mpc_exploration(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev, X_prev, n)
        t_expl[n] = time.perf_counter() - t0
        max_slack_matrix[n, :] = np.max(slack_vector)
        U_prev[:, :-1] = U_prev[:, 1:] 
        U_prev[:, -1] = U_prev[:, -1]
        X_prev[:, :-1] = X_prev[:, 1:] 
        X_prev[:, -1] = X_prev[:, -1] 

        # Find horizon until informative input
        h_star = get_informative_horizon(slack_vector, slack_tol)

        last_J_pess = J_pess[n, :] 
        last_J_opt = J_opt[n, :]         
        last_time_pess = t_pess[n] 
        last_time_opt = t_opt[n] 

        for h in range(h_star+1):
            if n >= N:
                break

            u_apply = U_mpc_exploration[:,h]
            U_applied.append(np.array(u_apply).squeeze())
            switched_logic[n, :] = 1

            y0, x0 = RNN_model_1step(x0, U_applied[-1], d_const)
            y_RNN.append(y0)

            # BLL
            x_real = (torch.from_numpy(np.array(x0))).to(torch.float32)
            y_real = (torch.from_numpy(np.array(y0))).to(torch.float32)
            noise = torch.normal(mean=0.0, std=np.sqrt(sigma2), size=y_real.shape)
            y_noisy = y_real + noise
            y_noisy = (torch.from_numpy(np.array(y_noisy))).to(torch.float32)
            phi_n = torch.cat([x_real.squeeze(), torch.tensor([1.0])])
            Lambda_inv, Q = recursive_update(phi_n, y_noisy, Lambda_inv, Q)
            theta_mean = Lambda_inv @ Q
            y_pred, y_std = predict(phi_n, theta_mean, Lambda_inv)
            y_BLL[n,:] = y_pred
            y_std_BLL[n, :] = y_std
            W_BLL[n, :] = theta_mean[:,0]
            theta_std = torch.sqrt(sigma2*torch.diag(Lambda_inv))
            W_std_BLL[n, :] = theta_std.numpy()
            Lambda_inv_store[n, :, :] = Lambda_inv
            J_pess[n, :] = last_J_pess
            J_opt[n, :] = last_J_opt
            t_opt[n] = last_time_opt
            t_pess[n] = last_time_pess

            t_total[n] = t_opt[n] + t_expl[n] 
            
            n += 1
    
    else:

        U_applied.append(np.array(U_mpc_pess).squeeze())
        switched_logic[n, :] = 0

        y0, x0 = RNN_model_1step(x0, U_applied[-1], d_const)
        y_RNN.append(y0)

        # BLL
        x_real = (torch.from_numpy(np.array(x0))).to(torch.float32)
        y_real = (torch.from_numpy(np.array(y0))).to(torch.float32)
        noise = torch.normal(mean=0.0, std=np.sqrt(sigma2), size=y_real.shape)
        y_noisy = y_real + noise
        y_noisy = (torch.from_numpy(np.array(y_noisy))).to(torch.float32)
        phi_n = torch.cat([x_real.squeeze(), torch.tensor([1.0])])
        Lambda_inv, Q = recursive_update(phi_n, y_noisy, Lambda_inv, Q)
        theta_mean = Lambda_inv @ Q
        y_pred, y_std = predict(phi_n, theta_mean, Lambda_inv)
        y_BLL[n,:] = y_pred
        y_std_BLL[n, :] = y_std
        W_BLL[n, :] = theta_mean[:,0]
        theta_std = torch.sqrt(sigma2*torch.diag(Lambda_inv))
        W_std_BLL[n, :] = theta_std.numpy()
        Lambda_inv_store[n, :, :] = Lambda_inv
        
        t_total[n] = t_opt[n] + t_expl[n] 

        n += 1



################################################ Plot ################################################

# Denormalize variables
U_applied = np.array(U_applied)
U_applied_den = (U_applied*input_scaler_scale[0][0]) + input_scaler_bias[0][0]
U_applied_den = (np.array(U_applied_den)).squeeze()

y_RNN_arr = np.array(y_RNN)  
y_RNN_arr = y_RNN_arr[:, :, 0]
y_RNN_den = (y_RNN_arr*output_scaler_scale) + output_scaler_bias

y_BLL_den = (y_BLL*output_scaler_scale) + output_scaler_bias
y_BLL_den = (np.array(y_BLL_den)).squeeze()

y_BLL_max_beta = y_BLL + beta_n * y_std_BLL
y_BLL_max_beta = (y_BLL_max_beta*output_scaler_scale) + output_scaler_bias
y_BLL_max_beta = (np.array(y_BLL_max_beta)).squeeze()

y_BLL_min_beta = y_BLL - beta_n * y_std_BLL
y_BLL_min_beta = (y_BLL_min_beta*output_scaler_scale) + output_scaler_bias
y_BLL_min_beta = (np.array(y_BLL_min_beta)).squeeze()

y_std_BLL = (np.array(y_std_BLL)).squeeze()

# Utils for plots
time = np.arange(0, N)
hour_ticks = np.array([0, 48, 96, 144, 192, 240, 288])
hour_labels = np.array([0, 4, 8, 12, 16, 20, 24]) 
pyplt.rcParams['mathtext.fontset'] = 'cm'  
pyplt.rcParams['font.family'] = 'serif'


# ========== PRICE ==========
pyplt.figure()
pyplt.plot(time, price/10, linewidth=5)
pyplt.ylabel(r'$c^{el}$ [€/kWh]', fontsize=60)
pyplt.xlim((0, N))
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()


# ========== INPUT PLOT ==========
pyplt.figure()
pyplt.plot(time, U_applied_den, linewidth=5)
pyplt.plot(time, 70*np.ones((N,1)), color='black', linewidth=5)
pyplt.plot(time, 90*np.ones((N,1)), color='black', linewidth=5)
pyplt.ylabel(r'$T_0^s$ [°C]', fontsize=60)
pyplt.ylim((70*0.99, 90*1.01))
pyplt.xlim((0, N))
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# ========== OUTPUT 1 + CONFIDENCE INTERVAL ==========
pyplt.figure()
pyplt.plot(time, y_RNN_den[:,0], linewidth=5, color='tab:blue')
pyplt.plot(time, y_BLL_den[:,0], linewidth=5, color='tab:orange')
pyplt.fill_between(time, y_BLL_min_beta[:,0], y_BLL_max_beta[:,0], color='orange', alpha=0.3)
pyplt.plot(time, y_min_T_variable[0:N], color='black', linewidth=5)
pyplt.plot(time, (y_max_T_value)*np.ones((N,1)), color='black', linewidth=5)
pyplt.ylabel(r'$T_5^s$ [°C]', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.ylim((y_min_T_variable.min()*0.99,y_max_T_value*1.000001))
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# # ZOOM
# fig, ax = pyplt.subplots()
# ax.plot(time, y_RNN_den[:,0], linewidth=5, color='tab:blue')
# ax.plot(time, y_BLL_den[:,0], linewidth=5, color='tab:orange')
# ax.fill_between(time,y_BLL_min_beta[:,0],y_BLL_max_beta[:,0],color='orange', alpha=0.3)
# ax.plot(time, y_min_T_variable[0:N], color='black', linewidth=5)
# ax.plot(time, y_max_T_value*np.ones((N,1)), color='black', linewidth=5)
# ax.set_ylabel(r'$T_5^s$ [°C]', fontsize=60)
# ax.set_xlabel(r'Time [h]', fontsize=60)
# ax.set_ylim((y_min_T_variable.min()*0.99,y_max_T_value*1.000001))
# ax.set_xlim((0, N))
# ax.set_xticks(hour_ticks)
# ax.set_xticklabels(hour_labels)
# ax.tick_params(axis='both', labelsize=60)
# # ================= INSET =================
# axins = zoomed_inset_axes(ax,zoom=2.5,loc='upper left',bbox_to_anchor=(0.2, 1.5),bbox_transform=ax.transAxes)
# axins.set_xticks([])
# axins.set_yticks([])
# axins.plot(time, y_RNN_den[:,0], linewidth=3, color='tab:blue')
# axins.plot(time, y_BLL_den[:,0], linewidth=3, color='tab:orange')
# axins.fill_between(time,y_BLL_min_beta[:,0],y_BLL_max_beta[:,0],color='orange', alpha=0.3)
# axins.set_xlim(0, 24)
# axins.set_ylim(70, 80)
# axins.tick_params(labelsize=24)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# pyplt.subplots_adjust(bottom=0.26, left=0.18)
# pyplt.show()

# ========== OUTPUT 2 + CONFIDENCE INTERVAL ==========
pyplt.figure()
pyplt.plot(time, y_RNN_den[:,1]/1000000, linewidth=5, color='tab:blue')
pyplt.plot(time, y_BLL_den[:,1]/1000000, linewidth=5, color='tab:orange')
pyplt.fill_between(time, y_BLL_min_beta[:,1]/1000000, y_BLL_max_beta[:,1]/1000000, color='orange', alpha=0.3)
pyplt.plot(time, y_min_P_value[0:N]/1000000, color='black', linewidth=5)
pyplt.plot(time, y_max_P_value/1000000*np.ones((N,1)), color='black', linewidth=5)
pyplt.ylabel(r'$P_0$ [MW]', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.ylim((y_min_P_value.min()/1000000*0.99, y_max_P_value/1000000*1.01))
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26, left=0.18)
pyplt.show()

# # ZOOM
# fig, ax = pyplt.subplots()
# ax.plot(time, y_RNN_den[:,1]/1000, linewidth=5, color='tab:blue')
# ax.plot(time, y_BLL_den[:,1]/1000, linewidth=5, color='tab:orange')
# ax.fill_between(time,
#                 y_BLL_min_beta[:,1]/1000,
#                 y_BLL_max_beta[:,1]/1000,
#                 color='orange', alpha=0.3)
# ax.plot(time, y_min_P_value[0:N]/1000, color='black', linewidth=5)
# ax.plot(time, y_max_P_value/1000*np.ones(N), color='black', linewidth=5)
# ax.set_ylabel(r'$P_0$ [kW]', fontsize=60)
# ax.set_xlabel(r'Time [h]', fontsize=60)
# ax.set_xlim((0, N))
# ax.set_ylim((y_min_P_value.min()/1000*0.99, y_max_P_value/1000*1.01))
# ax.set_xticks(hour_ticks)
# ax.set_xticklabels(hour_labels)
# ax.tick_params(axis='both', labelsize=60)
# # ================= INSET =================
# axins = zoomed_inset_axes(ax,zoom=2.5,loc='upper left',bbox_to_anchor=(0.2, 1.5),bbox_transform=ax.transAxes)
# axins.set_xticks([])
# axins.set_yticks([])
# axins.plot(time, y_RNN_den[:,1]/1000, linewidth=3, color='tab:blue')
# axins.plot(time, y_BLL_den[:,1]/1000, linewidth=3, color='tab:orange')
# axins.fill_between(time,y_BLL_min_beta[:,1]/1000,y_BLL_max_beta[:,1]/1000,color='orange', alpha=0.3)
# axins.set_xlim(0, 24)
# axins.set_ylim(750, 1500)
# axins.tick_params(labelsize=24)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# pyplt.subplots_adjust(bottom=0.26, left=0.18)
# pyplt.show()

# ========== COST FUNCTIONS ==========
pyplt.figure()
pyplt.plot(time, J_pess - J_opt, linewidth=5)
pyplt.plot(time, switching_threshold*np.ones((N,1)), linestyle='--', color='red', linewidth=5)
# Shade the regions where the cost difference is above/below threshold
above_threshold = (J_pess - J_opt).flatten() > switching_threshold
start_index = 0
current_state = above_threshold[0]
for i in range(1, N):
    if above_threshold[i] != current_state:
        pyplt.axvspan(start_index, i, 
                      facecolor='lightcoral' if current_state else 'lightblue',
                      alpha=0.3)
        start_index = i
        current_state = above_threshold[i]
# Shade the final region
pyplt.axvspan(start_index, N, 
              facecolor='lightcoral' if current_state else 'lightblue',
              alpha=0.3)
pyplt.ylabel('$J^p-J^o$', fontsize=60)
pyplt.xlabel('Time [h]', fontsize=60)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# ========== NORM OF WEIGHTS ==========
theta_star = np.zeros(n_theta)
theta_star[0:n_x] = Uo[0,0:n_x]
theta_star[n_x] = bo[0,0]
theta_star[n_x+1:2*(n_x+1)-1] = Uo[1,0:n_x]
theta_star[2*(n_x+1)-1] = bo[1,0]
theta_error = np.linalg.norm(W_BLL - theta_star, 2, axis=1)
pyplt.figure()
pyplt.plot(time, theta_error, linewidth=5)
pyplt.ylabel(r'$\|\theta - \theta^*\|_2$', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# ========== UNCERTAINTY ==========
pyplt.figure()
pyplt.plot(time, beta_n * y_std_BLL, linewidth=5)
pyplt.plot(time, epsilon_sigma * np.ones((N,1)), color='red', linestyle='--', linewidth=5)
pyplt.ylabel(r'$w$', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# ========== SLACK ==========
pyplt.figure()
pyplt.plot(time, max_slack_matrix, linewidth=5)
pyplt.ylabel(r'Slack', fontsize=60)
pyplt.xlim((0, N))
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()


################################################ Cost ################################################
P_real = y_RNN_den[:, 1]
price = np.asarray(price).reshape(-1)
J_economic = np.sum(price/10 * (P_real/1000) * 5/60)
print(f"Cost: {J_economic:.2f} €")


################################################ Final parameters ################################################
Uo_size = (Uo.flatten()).size
bo_size = (bo.flatten()).size
Uo_recovered = W_BLL[-1, :Uo_size]
bo_recovered = W_BLL[-1, Uo_size:Uo_size + bo_size]
print(f"Uo original: {Uo.flatten():}")
print(f"bo original: {bo.flatten():}")
Uo_rightshape = Uo_recovered.reshape((n_y, n_x))
bo_rightshape = bo_recovered.reshape((n_y, 1))
print(f"Uo right dimension: {Uo_rightshape:}")
print(f"bo right dimension: {bo_rightshape:}")

Lambda_0_last = Lambda_inv_store[-1, :, :]
print(f"Lambda: {Lambda_0_last:}")
np.save("Lambda_0_last.npy", Lambda_0_last)

print("Average MPC time per step: ", np.mean(t_total))
print("Maximum MPC time per step: ", np.max(t_total))


##################### Save results #####################
time = np.asarray(time).reshape(-1, 1)
U_applied_den = np.asarray(U_applied_den).reshape(-1, 1)
y_RNN_den = np.asarray(y_RNN_den)
y_BLL_den = np.asarray(y_BLL_den)
y_BLL_min_beta = np.asarray(y_BLL_min_beta)
y_BLL_max_beta = np.asarray(y_BLL_max_beta)
price = np.asarray(price).reshape(-1, 1)

data = {
    "time": time,
    "U_applied_den": U_applied_den,
    "y_RNN_den": y_RNN_den,
    "y_BLL_den": y_BLL_den,
    "y_BLL_min_beta": y_BLL_min_beta,
    "y_BLL_max_beta": y_BLL_max_beta,
    "y_min_T_variable": y_min_T_variable,
    "y_max_T_value": y_max_T_value,
    "y_min_P_value": y_min_P_value,
    "y_max_P_value": y_max_P_value,
    "price": price,
    "J_pess": J_pess,
    "J_opt": J_opt,
    "switching_threshold": switching_threshold,
    "max_slack_matrix": max_slack_matrix,
    "theta_error": theta_error
}

savemat("JpJo_solution.mat", data)