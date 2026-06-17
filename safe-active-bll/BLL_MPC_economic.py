#%% Import libraries
import casadi as ca
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as pyplt
from sklearn.metrics import mean_squared_error


################################################ Set values ################################################

# Read .mat file containing the NNARX weights and biases, extract parameters and scalers
file = scipy.io.loadmat('net_P.mat')
layers = file["layers"][0]
U1 = layers[0][0]["weights"][0][0]["U.0"][0]
W1 = layers[0][0]["weights"][0][0]["W.0"][0]
b1 = layers[0][0]["weights"][0][0]["b.0"][0]
Uo = layers[0][0]["weights"][0][0]["U0"][0]
bo = layers[0][0]["weights"][0][0]["b0"][0]
input_scaler_scale = file["input_scaler"][0]["scale"][0]
input_scaler_bias = file["input_scaler"][0]["bias"][0]
output_scaler_scale = file["output_scaler"][0]["scale"][0]
output_scaler_bias = file["output_scaler"][0]["bias"][0]

# MPC parameters
N = 288
n_u = 1
n_y = 1
n_x = 5
n_theta = n_x + n_y
H = 4
sigma2 = 0.001
epsilon_sigma = sigma2*1.01
epsilon_theta = sigma2*0.2
beta_n = 9
x_min = -5
x_max = 5
y_min = (10000 - output_scaler_bias) / output_scaler_scale
y_max = (3000000 - output_scaler_bias) / output_scaler_scale
u_min = (70 - input_scaler_bias) / input_scaler_scale
u_max = (90 - input_scaler_bias) / input_scaler_scale
delta = 0.0001
threshold = 0.01

price = np.ones(N)
price[0:72] = 1
price[72:144] = 2
price[144:216] = 1
price[216:288] = 2

P_ref = (540362 - output_scaler_bias)/output_scaler_scale #540362
u_ref = (80 - input_scaler_bias)/input_scaler_scale


################################################ Functions ################################################

# One step RNN
def RNN_model_1step_MPC(U_sequence, Y_0, h):
    # Initialization
    if h==0:
        u_km = U_sequence[:, h]
    else: 
        u_km = U_sequence[:, h-1]
    u_k = ca.vertcat(U_sequence[:, h])
    y_k = Y_0
    z1_k = ca.vertcat(y_k, u_km)
    # Prediction
    eta_1 = ca.tanh(U1.T@z1_k+W1.T@u_k+b1.T)
    #y_k = Uo.T@eta_1+bo.T     
    return eta_1


# One step RNN
def RNN_model_1step(U_sequence, Y_0):
    # Initialization
    if len(U_sequence) == 1:
        u_km = np.array([[U_sequence[-1]]])
    else: 
        u_km = np.array([[U_sequence[-2]]])
    u_k = np.array([[U_sequence[-1]]]) 
    y_k = Y_0 
    z1_k = np.concatenate([y_k, u_km], axis=0)
    # Prediction
    eta_1 = np.tanh(np.transpose(U1)@z1_k+np.transpose(W1)@u_k+np.transpose(b1))
    y_k = np.transpose(Uo)@eta_1+np.transpose(bo)    
    return y_k, eta_1


# Function to recursively update Lambda_inv and Q, as in (13)-(14) Pavone
def recursive_update(phi_t, y_t, Lambda_inv_prev, Q_prev):
    phi_t = phi_t.view(-1, 1)  
    y_t = y_t.view(-1, 1)     
    # Lambda_inv update
    denom = 1.0 + torch.matmul(phi_t.T, Lambda_inv_prev @ phi_t)
    Lambda_inv_new = Lambda_inv_prev - ((Lambda_inv_prev @ phi_t) @ (Lambda_inv_prev @ phi_t).T) / denom
    # Q update
    Q_new = phi_t @ y_t.T + Q_prev 
    return Lambda_inv_new, Q_new


# Function to compute y predicted by BLL
def predict(phi_t, theta_mean, Lambda_inv):
    y_pred = theta_mean.T @ phi_t 
    y_std = torch.sqrt((1 + phi_t.T @ Lambda_inv @ phi_t)*sigma2)
    xT_L_x = phi_t.T @ Lambda_inv @ phi_t
    return y_pred, y_std, xT_L_x


# MPC for pessimistic exploration
def mpc_exploration_pessimistic(x_n, Lambda_inv_n, Q_n, theta_mean_n, theta_std_n, y_pred_n, y_std_n, U_prev, X_prev, n):

    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    lower_bound = opti.variable(n_y, H)
    upper_bound = opti.variable(n_y, H)
    Sigma_y = opti.variable(n_y, H)
    # slack_sigma = opti.variable(n_y, H)

    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    y_mean_prev = ca.MX(y_pred_n.detach().cpu().numpy())
    y_std_prev = ca.MX(y_std_n.detach().cpu().numpy())
    l_prev = y_mean_prev - ca.sqrt(beta_n) * y_std_prev
    u_prev = y_mean_prev + ca.sqrt(beta_n) * y_std_prev
    if n+H <= N:
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    current_price_ca = ca.DM(current_price)
    P_ref_ca = ca.DM(P_ref)
    U_ref_ca = ca.DM(u_ref)

    # Initial values
    opti.subject_to(X[:,0] == x_n)
    opti.subject_to(lower_bound[:,0] == l_prev)
    opti.subject_to(upper_bound[:,0] == u_prev)

    # # Slack variables non negativity
    # opti.subject_to(slack_sigma >= 0)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # RNN model
        x_pred_RNN = RNN_model_1step_MPC(U, ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)), h)
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Uncertainty constraint
        opti.subject_to(Sigma_y[:,h] == sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))
        # opti.subject_to(Sigma_y[:,h] >= epsilon_sigma - slack_sigma[:,h])

        # Pessimistic set constraint
        if h > 0:
            opti.subject_to(lower_bound[:,h] == ca.fmax(lower_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(upper_bound[:,h] == ca.fmin(upper_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(lower_bound[:,h] >= y_min)
            opti.subject_to(upper_bound[:,h] <= y_max)
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])

        # Cost function
        J_step += 1*current_price_ca[h] * ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)))
        #J_step += 0.001*ca.sumsqr(U[:,h] - U_ref_ca)
        #J_step += U[:,h]
        J_step -= 1000*ca.sumsqr(Sigma_y[:,h]) #1000
    
    # terminal_price = ca.DM(price[n + H])
    # J_step += terminal_price * ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,H], 1)))
    #J_step += 10*ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,H], 1)) - P_ref_ca)
    #J_step += 100000*ca.sumsqr(U[:,H-1] - U_ref_ca)
    #opti.subject_to(U[:, H-1] >= U[:, 0])

    # Minimize
    opti.minimize(J_step)
    #opti.minimize(10 * ca.sum(slack_sigma) + J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)) )
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)) )
    # opti.set_initial(slack_sigma, 0)
    opti.set_initial(Sigma_y, np.tile(np.array(y_std_n).reshape(-1, 1), (1, H)))

    # Solve problem
    opti.solver('ipopt', {"ipopt.max_iter": 5000})
    sol = opti.solve()
    U_opt = sol.value(U[:,0])
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)
    # slack_max = np.max(sol.value(slack_sigma))
    # slack_first = sol.value(slack_sigma[:,0])
    slack_max = 0
    slack_first = 0

    return U_opt, slack_max, slack_first, U_opt_tot, X_opt_tot


# Pessimistic MPC
def mpc_pessimistic(x_n, Lambda_inv_n, theta_mean_n, theta_std_n, y_pred_n, y_std_n, U_prev, X_prev, switched, n):
    
    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    lower_bound = opti.variable(n_y, H)
    upper_bound = opti.variable(n_y, H)
    #J = opti.variable(1, H)
    
    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    y_mean_prev = ca.MX(y_pred_n.detach().cpu().numpy())
    y_std_prev = ca.MX(y_std_n.detach().cpu().numpy())
    l_prev = y_mean_prev - ca.sqrt(beta_n) * y_std_prev
    u_prev = y_mean_prev + ca.sqrt(beta_n) * y_std_prev
    if n+H <= N:
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    current_price_ca = ca.DM(current_price)
    P_ref_ca = ca.DM(P_ref)

    # Initial values
    opti.subject_to(X[:,0] == x_n)
    opti.subject_to(lower_bound[:,0] == l_prev)
    opti.subject_to(upper_bound[:,0] == u_prev)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # RNN model
        x_pred_RNN = RNN_model_1step_MPC(U, ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)), h)
        opti.subject_to(X[:,h+1] == x_pred_RNN)
        
        # Pessimistic set constraint
        if h > 0:
            opti.subject_to(lower_bound[:,h] == ca.fmax(lower_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(upper_bound[:,h] == ca.fmin(upper_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(lower_bound[:,h] >= y_min)
            opti.subject_to(upper_bound[:,h] <= y_max)
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])

        # Cost function
        J_step += current_price_ca[h] * ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)))
        # if switched:
        #     J_step += 0.0001*ca.sumsqr(U[:,h])
        
    # Minimize
    opti.minimize(J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)))

    # Solve problem
    opti.solver('ipopt', {"ipopt.max_iter": 5000})
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
    
    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    theta_std_prev = ca.MX(theta_std_n.detach().cpu().numpy())
    y_mean_prev = ca.MX(y_pred_n.detach().cpu().numpy())
    y_std_prev = ca.MX(y_std_n.detach().cpu().numpy())
    l_prev = y_mean_prev - ca.sqrt(beta_n) * y_std_prev
    u_prev = y_mean_prev + ca.sqrt(beta_n) * y_std_prev
    if n+H <= N:
        current_price = price[n:n+H]
    else:
        dim_left = H - N + n
        current_price = np.concatenate([price[n:], np.full(dim_left, price[-1])])
    current_price_ca = ca.DM(current_price)
    P_ref_ca = ca.DM(P_ref)

    # Initial values
    opti.subject_to(X[:,0] == x_n)
    opti.subject_to(lower_bound[:,0] == l_prev)
    opti.subject_to(upper_bound[:,0] == u_prev)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):
        # RNN model
        x_pred_RNN = RNN_model_1step_MPC(U, ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)), h)
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Optimistic set constraint
        if h > 0:
            opti.subject_to(lower_bound[:,h] == ca.fmax(lower_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(upper_bound[:,h] == ca.fmin(upper_bound[:,h-1], ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + ca.sqrt(beta_n) * ca.sqrt(sigma2 * (1 + ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)])))))
            opti.subject_to(lower_bound[:,h] <= y_max-delta)
            opti.subject_to(upper_bound[:,h] >= y_min+delta)
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])       
        
        # Constraint on y
        opti.subject_to(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)) <= upper_bound[:,h])
        opti.subject_to(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)) >= lower_bound[:,h])
    
        # Cost function
        J_step += current_price_ca[h] * ca.sumsqr(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)))

    # Minimize
    opti.minimize(J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + np.sqrt(beta_n) * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(Theta, np.tile(np.array(theta_mean_n).reshape(-1, 1), (1, H)))

    # Solve problem
    opti.solver('ipopt', {"ipopt.max_iter": 5000})
    sol = opti.solve()
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)

    return sol.value(J_step), U_opt_tot, X_opt_tot


################################################ Simulation ################################################

# MPC initialization
x0 = np.random.uniform(-1, 1, size=n_x)
U_prev = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_pess = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev_pess = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_opt = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev_opt = 2 * np.random.rand(n_x, H + 1) - 1

U_applied = []
y_RNN = []
U_explo = []
U_pess = []

# BNN prior
Lambda_inv = torch.eye(n_theta) 
Q = torch.zeros((n_theta, n_y))  
theta_mean = Lambda_inv @ Q
theta_std = torch.sqrt(sigma2*torch.diag(Lambda_inv))
x0_BNN = (torch.from_numpy(np.array(x0))).to(torch.float32)
y_pred, y_std, xT_L_x = predict(torch.cat([x0_BNN.squeeze(), torch.tensor([1.0])]), theta_mean, Lambda_inv)
y_BLL = np.zeros((N, n_y)) 
y_std_BLL = np.zeros((N, n_y)) 
W_BLL = np.zeros((N, n_theta)) 
W_std_BLL = np.zeros((N, n_theta)) 
J_pess = np.zeros((N, 1)) 
J_opt = np.zeros((N, 1)) 
slack_max = np.zeros((N, 1))
slack_first = np.zeros((N, 1))
xT_L_x_all = np.zeros((N, 1))
switched_logic = np.zeros((N, 1))
switched = False

for n in range(N):
    # Exploration
    U_mpc_exploration, slack_max[n,:], slack_first[n,:], U_prev, X_prev = mpc_exploration_pessimistic(x0, Lambda_inv, Q, theta_mean, theta_std, y_pred, y_std, U_prev, X_prev, n)
    U_explo.append(U_mpc_exploration)
    # Pessimistic
    U_mpc_pess, J_pess[n,:], U_prev_pess, X_prev_pess = mpc_pessimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_pess, X_prev_pess, switched, n)
    U_pess.append(U_mpc_pess)
    # Optimistic
    J_opt[n,:], U_prev_opt, X_prev_opt = mpc_optimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_opt, X_prev_opt, n)

    if not switched:
        # # Exploration
        # U_mpc_exploration, slack_max[n,:], slack_first[n,:], U_prev, X_prev = mpc_exploration_pessimistic(x0, Lambda_inv, Q, theta_mean, y_pred, y_std, U_prev, X_prev)
        # U_explo.append(U_mpc_exploration)
        # # Optimistic
        # J_opt[n,:], U_prev_opt, X_prev_opt = mpc_optimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_opt, X_prev_opt)
        U_applied.append(U_mpc_exploration)
        switched_logic[n, :] = 0
    else:
        U_applied.append(U_mpc_pess)
        switched_logic[n, :] = 0.5

    #if not switched and n>5 and (J_pess[n] - J_opt[n] <= threshold) and (J_pess[n-1] - J_opt[n-1] <= threshold) and (J_pess[n-2] - J_opt[n-2] <= threshold) and (J_pess[n-3] - J_opt[n-3] <= threshold) and (J_pess[n-4] - J_opt[n-4] <= threshold) and (J_pess[n-5] - J_opt[n-5] <= threshold):
    if not switched and n >=144:
        switched = True  
    
    if n == 0:
        y0 = np.array(P_ref).reshape(1,1)
    
    y0, x0 = RNN_model_1step(U_applied, y0)
    y_RNN.append(y0)
    U_prev[:, :-1] = U_prev[:, 1:] 
    U_prev[:, -1] = U_prev[:, -1]
    X_prev[:, :-1] = X_prev[:, 1:] 
    X_prev[:, -1] = X_prev[:, -1]
    U_prev_pess[:, :-1] = U_prev_pess[:, 1:] 
    U_prev_pess[:, -1] = U_prev_pess[:, -1]
    X_prev_pess[:, :-1] = X_prev_pess[:, 1:] 
    X_prev_pess[:, -1] = X_prev_pess[:, -1]
    U_prev_opt[:, :-1] = U_prev_opt[:, 1:] 
    U_prev_opt[:, -1] = U_prev_opt[:, -1]
    X_prev_opt[:, :-1] = X_prev_opt[:, 1:] 
    X_prev_opt[:, -1] = X_prev_opt[:, -1]

    # BLL
    x_real = (torch.from_numpy(np.array(x0))).to(torch.float32)
    y_real = (torch.from_numpy(np.array(y0))).to(torch.float32)
    y_noisy = y_real + torch.randn_like(y_real)*np.sqrt(sigma2)
    y_noisy = (torch.from_numpy(np.array(y_noisy))).to(torch.float32)
    phi_n = torch.cat([x_real.squeeze(), torch.tensor([1.0])])
    Lambda_inv, Q = recursive_update(phi_n, y_noisy, Lambda_inv, Q)
    theta_mean = Lambda_inv @ Q
    y_pred, y_std, xT_L_x = predict(phi_n, theta_mean, Lambda_inv)
    y_BLL[n,:] = y_pred
    y_std_BLL[n, :] = y_std
    W_BLL[n, :] = theta_mean[:,0]
    theta_std = torch.sqrt(sigma2*torch.diag(Lambda_inv))
    W_std_BLL[n, :] = theta_std.numpy()
    xT_L_x_all[n,:] = xT_L_x



################################################ Plot ################################################

U_applied_den = (U_applied*input_scaler_scale) + input_scaler_bias
U_applied_den = (np.array(U_applied_den)).squeeze()
U_explo = (U_explo*input_scaler_scale) + input_scaler_bias
U_explo = (np.array(U_explo)).squeeze()
U_pess = (U_pess*input_scaler_scale) + input_scaler_bias
U_pess = (np.array(U_pess)).squeeze()

y_RNN_den = (y_RNN*output_scaler_scale) + output_scaler_bias
y_RNN_den = (np.array(y_RNN_den)).squeeze()

y_BLL_den = (y_BLL*output_scaler_scale) + output_scaler_bias
y_BLL_den = (np.array(y_BLL_den)).squeeze()

y_BLL_max = y_BLL + 1.96 * y_std_BLL
y_BLL_max = (y_BLL_max*output_scaler_scale) + output_scaler_bias
y_BLL_max = (np.array(y_BLL_max)).squeeze()
y_BLL_max_beta = y_BLL + np.sqrt(beta_n) * y_std_BLL
y_BLL_max_beta = (y_BLL_max_beta*output_scaler_scale) + output_scaler_bias
y_BLL_max_beta = (np.array(y_BLL_max_beta)).squeeze()

y_BLL_min = y_BLL - 1.96 * y_std_BLL
y_BLL_min = (y_BLL_min*output_scaler_scale) + output_scaler_bias
y_BLL_min = (np.array(y_BLL_min)).squeeze()
y_BLL_min_beta = y_BLL - np.sqrt(beta_n) * y_std_BLL
y_BLL_min_beta = (y_BLL_min_beta*output_scaler_scale) + output_scaler_bias
y_BLL_min_beta = (np.array(y_BLL_min_beta)).squeeze()

y_std_BLL = (np.array(y_std_BLL)).squeeze()

mse_y = mean_squared_error([y_RNN_den[-1]], [y_BLL_den[-1]])
mse_Theta0 = mean_squared_error(Uo[0], [W_BLL[-1,0]])
mse_Theta1 = mean_squared_error(Uo[1], [W_BLL[-1,1]])
mse_Theta2 = mean_squared_error(Uo[2], [W_BLL[-1,2]])
mse_Theta3 = mean_squared_error(Uo[3], [W_BLL[-1,3]])
mse_Theta4 = mean_squared_error(Uo[4], [W_BLL[-1,4]])
mse_Theta5 = mean_squared_error(bo, [W_BLL[-1,5]])

fig, ax = pyplt.subplots(4, 1)
time = np.arange(0, N)
ax[0].plot(time, U_applied_den)
ax[0].set_ylabel('Input [°C]', fontsize=10)
ax[0].set_xlim((0, N))

ax[1].plot(time, y_RNN_den, label='RNN')
ax[1].plot(time, y_BLL_den, label=f'BLL (MSE ={mse_y})')
#ax[1].plot(time, Y_ref_den*np.ones((N,1)), color='black', label='Y_ref')
ax[1].fill_between(time, y_BLL_min_beta, y_BLL_max_beta, color='orange', alpha=0.3, label='Beta CI')
ax[1].fill_between(time, y_BLL_min, y_BLL_max, color='green', alpha=0.3, label='95% CI')
ax[1].legend(fontsize=6)
ax[1].set_ylabel('Output [°C]', fontsize=10)
ax[1].set_xlim((0, N))

ax[2].plot(time, y_std_BLL**2)
ax[2].plot(time, epsilon_sigma*np.ones((N,1)), color='black', label='epsilon')
ax[2].legend(fontsize=6)
ax[2].set_ylabel('Sigma^2', fontsize=10)
ax[2].set_xlim((0, N))

ax[3].plot(time, (W_std_BLL[:, 0])**2)
ax[3].plot(time, (W_std_BLL[:, 1])**2)
ax[3].plot(time, (W_std_BLL[:, 2])**2)
ax[3].plot(time, (W_std_BLL[:, 3])**2)
ax[3].plot(time, (W_std_BLL[:, 4])**2)
ax[3].plot(time, (W_std_BLL[:, 5])**2)
ax[3].plot(time, epsilon_theta*np.ones((N,1)), color='black', label='epsilon')
ax[3].legend(fontsize=6)
ax[3].set_ylabel('Theta_var', fontsize=10)
ax[3].set_xlim((0, N))
pyplt.show()

fig, ax = pyplt.subplots(6, 1)
time = np.arange(0, N)
ax[0].plot(time, Uo[0]*np.ones((N,1)), color='black')
ax[0].plot(time, W_BLL[:,0], label=f'MSE ={mse_Theta0}')
ax[0].fill_between(time, W_BLL[:, 0] - np.sqrt(beta_n) * W_std_BLL[:, 0], W_BLL[:, 0] + np.sqrt(beta_n) * W_std_BLL[:, 0], color='orange', alpha=0.3)
ax[0].fill_between(time, W_BLL[:, 0] - 1.96 * W_std_BLL[:, 0], W_BLL[:, 0] + 1.96 * W_std_BLL[:, 0], color='green', alpha=0.3)
ax[0].set_ylabel('Weight', fontsize=10)
ax[0].legend(fontsize=6)
ax[0].set_xlim((0, N))

ax[1].plot(time, Uo[1]*np.ones((N,1)), color='black')
ax[1].plot(time, W_BLL[:,1], label=f'MSE ={mse_Theta1}')
ax[1].fill_between(time, W_BLL[:, 1] - np.sqrt(beta_n) * W_std_BLL[:, 1], W_BLL[:, 1] + np.sqrt(beta_n) * W_std_BLL[:, 1], color='orange', alpha=0.3)
ax[1].fill_between(time, W_BLL[:, 1] - 1.96 * W_std_BLL[:, 1], W_BLL[:, 1] + 1.96 * W_std_BLL[:, 1], color='green', alpha=0.3)
ax[1].set_ylabel('Weight', fontsize=10)
ax[1].legend(fontsize=6)
ax[1].set_xlim((0, N))

ax[2].plot(time, Uo[2]*np.ones((N,1)), color='black')
ax[2].plot(time, W_BLL[:,2], label=f'MSE ={mse_Theta2}')
ax[2].fill_between(time, W_BLL[:, 2] - np.sqrt(beta_n) * W_std_BLL[:, 2], W_BLL[:, 2] + np.sqrt(beta_n) * W_std_BLL[:, 2], color='orange', alpha=0.3)
ax[2].fill_between(time, W_BLL[:, 2] - 1.96 * W_std_BLL[:, 2], W_BLL[:, 2] + 1.96 * W_std_BLL[:, 2], color='green', alpha=0.3)
ax[2].set_ylabel('Weight', fontsize=10)
ax[2].legend(fontsize=6)
ax[2].set_xlim((0, N))

ax[3].plot(time, Uo[3]*np.ones((N,1)), color='black')
ax[3].plot(time, W_BLL[:,3], label=f'MSE ={mse_Theta3}')
ax[3].fill_between(time, W_BLL[:, 3] - np.sqrt(beta_n) * W_std_BLL[:, 3], W_BLL[:, 3] + np.sqrt(beta_n) * W_std_BLL[:, 3], color='orange', alpha=0.3)
ax[3].fill_between(time, W_BLL[:, 3] - 1.96 * W_std_BLL[:, 3], W_BLL[:, 3] + 1.96 * W_std_BLL[:, 3], color='green', alpha=0.3)
ax[3].set_ylabel('Weight', fontsize=10)
ax[3].legend(fontsize=6)
ax[3].set_xlim((0, N))

ax[4].plot(time, Uo[4]*np.ones((N,1)), color='black')
ax[4].plot(time, W_BLL[:,4], label=f'MSE ={mse_Theta4}')
ax[4].fill_between(time, W_BLL[:, 4] - np.sqrt(beta_n) * W_std_BLL[:, 4], W_BLL[:, 4] + np.sqrt(beta_n) * W_std_BLL[:, 4], color='orange', alpha=0.3)
ax[4].fill_between(time, W_BLL[:, 4] - 1.96 * W_std_BLL[:, 4], W_BLL[:, 4] + 1.96 * W_std_BLL[:, 4], color='green', alpha=0.3)
ax[4].set_ylabel('Weight', fontsize=10)
ax[4].legend(fontsize=6)
ax[4].set_xlim((0, N))

ax[5].plot(time, bo*np.ones((N,1)), color='black', label='RNN')
ax[5].plot(time, W_BLL[:,5], label=f'BLL, MSE ={mse_Theta0}')
ax[5].fill_between(time, W_BLL[:, 5] - np.sqrt(beta_n) * W_std_BLL[:, 5], W_BLL[:, 5] + np.sqrt(beta_n) * W_std_BLL[:, 5], color='orange', alpha=0.3, label='Beta CI')
ax[5].fill_between(time, W_BLL[:, 5] - 1.96 * W_std_BLL[:, 5], W_BLL[:, 5] + 1.96 * W_std_BLL[:, 5], color='green', alpha=0.3, label='95% CI')
ax[5].legend(fontsize=6)
ax[5].set_ylabel('Bias', fontsize=10)
ax[5].set_xlim((0, N))
pyplt.show()

pyplt.plot(time, J_pess, label='J_pess')
pyplt.plot(time, J_opt, label='J_opt')
pyplt.plot(time, J_pess-J_opt, label='J_pess-J_opt')
pyplt.plot(time, switched_logic, label='Switched logic')
pyplt.legend(fontsize=6)
pyplt.xlim((0, N))
pyplt.show()

# pyplt.plot(time, xT_L_x_all, label="xT Lambda_inv x")
# pyplt.legend(fontsize=6)
# pyplt.xlim((0, N))
# pyplt.show()

# pyplt.plot(time, U_pess, label='U_pess')
# pyplt.plot(time, U_explo, label='U_explo')
# pyplt.plot(time, U_applied_den, label='U_applied')
# pyplt.legend(fontsize=6)
# pyplt.ylabel('J', fontsize=10)
# pyplt.xlim((0, N))
# pyplt.show()
