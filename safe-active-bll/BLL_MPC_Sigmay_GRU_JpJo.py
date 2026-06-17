#%% Import libraries
import casadi as ca
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as pyplt
from sklearn.metrics import mean_squared_error


################################################ Set values ################################################

# MPC parameters
N = 288
n_u = 1
n_y = 1
n_x = 5
n_theta = n_x + 1
H = 12
sigma2 = 0.001
beta_n = 9
coeff = 1.5
epsilon_sigma = np.sqrt(sigma2)*coeff
x_min = -1
x_max = 1
delta = 0.0001
Lips=0.2

# Read .mat file containing the GRU weights and biases, extract parameters and scalers
file = scipy.io.loadmat('net_GRU_5.mat')
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

y_min = (70 - output_scaler_bias) / output_scaler_scale
y_max = (90 - output_scaler_bias) / output_scaler_scale
u_min = (70 - input_scaler_bias) / input_scaler_scale
u_max = (90 - input_scaler_bias) / input_scaler_scale
Y_ref_den = np.ones(N)
Y_ref_den[0:96] = 88
Y_ref_den[96:192] = 74
Y_ref_den[192:] = 85 
Y_ref_vec = (Y_ref_den - output_scaler_bias) / output_scaler_scale


################################################ Functions ################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# One step GRU for MPC
def RNN_model_1step_MPC(xk, uk):
    uk = ca.reshape(uk, (n_u, 1)) 
    xk = ca.reshape(xk, (n_x, 1))
    zk = sigmoid(Wz@uk + Uz@xk + bz)
    fk = sigmoid(Wf@uk + Uf@xk + bf)
    phi = ca.tanh(Wr@uk + Ur@(fk*xk) + br)
    xkp = zk * xk + (1-zk) * phi    
    return xkp

# One step GRU
def RNN_model_1step(xk, uk):
    uk = np.atleast_1d(uk).reshape(n_u, 1) 
    xk = xk.reshape(n_x, 1) 
    zk = sigmoid(Wz@uk + Uz@xk + bz)
    fk = sigmoid(Wf@uk + Uf@xk + bf)
    phi = np.tanh(Wr@uk + Ur@(fk*xk) + br)
    xkp = zk * xk + (1-zk) * phi
    ykp = Uo @ xkp + bo 
    return ykp, xkp

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
    y_std = torch.sqrt((phi_t.T @ Lambda_inv @ phi_t)*sigma2)
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
    slack_sigma = opti.variable(n_y, H)

    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    if n+H <= N:
        Y_ref = Y_ref_vec[:,n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.concatenate([Y_ref_vec[:,n:], np.full((1,dim_left), Y_ref_vec[:,-1])], axis=1)
    Y_ref = Y_ref.reshape((n_y, H))
    Y_ref_ca = ca.DM(Y_ref)

    # Initial values
    opti.subject_to(X[:,0] == x_n)
    
    # Slack variables non negativity
    opti.subject_to(slack_sigma >= 0)

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
        x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h])
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Uncertainty constraint
        opti.subject_to(Sigma_y[:,h] == ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(beta_n*Sigma_y[:,h] >= epsilon_sigma - slack_sigma[:,h])

        # Pessimistic set constraint
        opti.subject_to(lower_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(upper_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])
        opti.subject_to(lower_bound[:,h] >= y_min)
        opti.subject_to(upper_bound[:,h] <= y_max)

        # Cost function
        J_step += 10 * ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - Y_ref_ca[:,h])

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])

    # Minimize
    opti.minimize(100 * ca.sum(slack_sigma) + J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)) )
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)) )
    opti.set_initial(slack_sigma, 0)
    opti.set_initial(Sigma_y, np.tile(np.array(y_std_n).reshape(-1, 1), (1, H)))

    # Solve problem
    opti.solver('ipopt', {"ipopt.max_iter": 5000})
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
    Sigma_y = opti.variable(n_y, H)
    
    # Needed variables
    Lambda_inv_prev = ca.MX(Lambda_inv_n.detach().cpu().numpy())  
    theta_mean_prev = ca.MX(theta_mean_n.detach().cpu().numpy())
    if n+H <= N:
        Y_ref = Y_ref_vec[:,n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.concatenate([Y_ref_vec[:,n:], np.full((1,dim_left), Y_ref_vec[:,-1])], axis=1)
    Y_ref = Y_ref.reshape((1, H))
    Y_ref_ca = ca.DM(Y_ref)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):

        # RNN model
        x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h])
        opti.subject_to(X[:,h+1] == x_pred_RNN)

         # Uncertainty constraint
        opti.subject_to(Sigma_y[:,h] == ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))

        # Pessimistic set constraint
        opti.subject_to(lower_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(upper_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(lower_bound[:,h] >= y_min)
        opti.subject_to(upper_bound[:,h] <= y_max)
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])

        # Cost function
        if not switched:
            J_step += ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - Y_ref_ca[:,h]) + ca.sum1(Lips*beta_n*Sigma_y[:,h])
        else:
            J_step += ca.sumsqr(ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - Y_ref_ca[:,h]) + 0.01*ca.sumsqr(U[:,h])

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])

    # Minimize
    opti.minimize(J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)))

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
    if n+H <= N:
        Y_ref = Y_ref_vec[:,n:n+H]
    else:
        dim_left = H - N + n
        Y_ref = np.concatenate([Y_ref_vec[:,n:], np.full((1,dim_left), Y_ref_vec[:,-1])], axis=1)
    Y_ref = Y_ref.reshape((1, H))
    Y_ref_ca = ca.DM(Y_ref)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Loop over prediction horizon
    J_step = 0
    for h in range(H):
        # RNN model
        x_pred_RNN = RNN_model_1step_MPC(X[:,h], U[:,h])
        opti.subject_to(X[:,h+1] == x_pred_RNN)

        # Optimistic set constraint
        opti.subject_to(lower_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) - beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(upper_bound[:,h] == ca.mtimes(theta_mean_prev.T, ca.vertcat(X[:,h], 1)) + beta_n * ca.sqrt(sigma2 * (ca.mtimes([ca.vertcat(X[:,h], 1).T, Lambda_inv_prev, ca.vertcat(X[:,h], 1)]))))
        opti.subject_to(lower_bound[:,h] <= upper_bound[:,h])      
        if h == 0 or h == H-1: # invariant set
            opti.subject_to(lower_bound[:,h] >= y_min)
            opti.subject_to(upper_bound[:,h] <= y_max)
        else: # optimistic set
            opti.subject_to(lower_bound[:,h] <= y_max-delta)
            opti.subject_to(upper_bound[:,h] >= y_min+delta) 
        
        # Constraint on y
        opti.subject_to(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)) <= upper_bound[:,h])
        opti.subject_to(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)) >= lower_bound[:,h])

        # Cost function
        J_step += ca.sumsqr(ca.mtimes(Theta[:,h].T, ca.vertcat(X[:,h], 1)) - Y_ref_ca[:,h])

    # Terminal set
    #opti.subject_to(X[:,H]==X[:,H-1])

    # Minimize
    opti.minimize(J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(lower_bound, np.tile(np.array((y_pred_n - beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(upper_bound, np.tile(np.array((y_pred_n + beta_n * y_std_n)).reshape(-1, 1), (1, H)))
    opti.set_initial(Theta, np.tile(np.array(theta_mean_n).reshape(-1, 1), (1, H)))

    # Solve problem
    opti.solver('ipopt', {"ipopt.max_iter": 5000})
    sol = opti.solve()
    U_opt_tot = sol.value(U).reshape((n_u, H))
    X_opt_tot = sol.value(X)

    return sol.value(J_step), U_opt_tot, X_opt_tot


def get_informative_horizon(slack_seq):
    slack_seq = np.atleast_2d(slack_seq)
    H = slack_seq.shape[1]
    for h in range(H):
        if np.allclose(slack_seq[:, h], 0.0, atol=1e-6):
            return h
    return H-1 


################################################ Simulation ################################################

# MPC initialization
np.random.seed(3) # choose always the same random numbers
x0 = np.random.uniform(-1, 1, size=n_x)
U_prev = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_pess = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev_pess = 2 * np.random.rand(n_x, H + 1) - 1
U_prev_opt = np.ones((n_u, H))*((70 - input_scaler_bias) / input_scaler_scale)
X_prev_opt = 2 * np.random.rand(n_x, H + 1) - 1

U_applied = []
y_RNN = []

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
slack_vector = np.zeros((1, H))
slack_matrix = np.zeros((N, H))
xT_L_x_all = np.zeros((N, 1))

switched_logic = np.zeros((N, 1))
switched = False
n = 0

while n < N:
    # Pessimistic
    U_mpc_pess, J_pess[n,:], U_prev_pess, X_prev_pess = mpc_pessimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_pess, X_prev_pess, switched, n)
    # Optimistic
    J_opt[n,:], U_prev_opt, X_prev_opt = mpc_optimistic(x0, Lambda_inv, theta_mean, theta_std, y_pred, y_std, U_prev_opt, X_prev_opt, n)
       
    U_prev_pess[:, :-1] = U_prev_pess[:, 1:] 
    U_prev_pess[:, -1] = U_prev_pess[:, -1]
    X_prev_pess[:, :-1] = X_prev_pess[:, 1:] 
    X_prev_pess[:, -1] = X_prev_pess[:, -1]
    U_prev_opt[:, :-1] = U_prev_opt[:, 1:] 
    U_prev_opt[:, -1] = U_prev_opt[:, -1]
    X_prev_opt[:, :-1] = X_prev_opt[:, 1:] 
    X_prev_opt[:, -1] = X_prev_opt[:, -1]

    # Switch logic
    if not switched and (J_pess[n,:] -J_opt[n,:] <= 2*epsilon_sigma*H*Lips): #np.sqrt(sigma2)*beta_n*H*coeff) : # 0.05
        switched = True  

    if switched and (J_pess[n,:] -J_opt[n,:] > 2*epsilon_sigma*H*Lips) :
        switched = False  

    if not switched:
        # Exploration
        U_mpc_exploration, slack_vector, U_prev, X_prev = mpc_exploration_pessimistic(x0, Lambda_inv, Q, theta_mean, theta_std, y_pred, y_std, U_prev, X_prev, n)
        slack_matrix[n,:] = slack_vector
        U_prev[:, :-1] = U_prev[:, 1:] 
        U_prev[:, -1] = U_prev[:, -1]
        X_prev[:, :-1] = X_prev[:, 1:] 
        X_prev[:, -1] = X_prev[:, -1] 

        # Find horizon until informative input
        h_star = get_informative_horizon(slack_vector)

        last_J_pess = J_pess[n, :] 
        last_J_opt = J_opt[n, :] 

        for h in range(h_star+1):
            if n >= N:
                break

            u_apply = U_mpc_exploration[:,h]
            U_applied.append(np.array(u_apply).squeeze())
            switched_logic[n, :] = 0

            y0, x0 = RNN_model_1step(x0, U_applied[-1])
            y_RNN.append(y0)

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

            J_pess[n, :] = last_J_pess
            J_opt[n, :] = last_J_opt

            n += 1
    
    else:

        U_applied.append(np.array(U_mpc_pess).squeeze())
        switched_logic[n, :] = 1

        y0, x0 = RNN_model_1step(x0, U_applied[-1])
        y_RNN.append(y0)

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

        n += 1



################################################ Plot ################################################
U_applied = np.array(U_applied)
U_applied_den = (U_applied*input_scaler_scale) + input_scaler_bias
U_applied_den = (np.array(U_applied_den)).squeeze()

y_RNN_den = (y_RNN*output_scaler_scale) + output_scaler_bias
y_RNN_den = (np.array(y_RNN_den)).squeeze()

y_BLL_den = (y_BLL*output_scaler_scale) + output_scaler_bias
y_BLL_den = (np.array(y_BLL_den)).squeeze()

y_BLL_max = y_BLL + 1.96 * y_std_BLL
y_BLL_max = (y_BLL_max*output_scaler_scale) + output_scaler_bias
y_BLL_max = (np.array(y_BLL_max)).squeeze()
y_BLL_max_beta = y_BLL + beta_n * y_std_BLL
y_BLL_max_beta = (y_BLL_max_beta*output_scaler_scale) + output_scaler_bias
y_BLL_max_beta = (np.array(y_BLL_max_beta)).squeeze()

y_BLL_min = y_BLL - 1.96 * y_std_BLL
y_BLL_min = (y_BLL_min*output_scaler_scale) + output_scaler_bias
y_BLL_min = (np.array(y_BLL_min)).squeeze()
y_BLL_min_beta = y_BLL - beta_n * y_std_BLL
y_BLL_min_beta = (y_BLL_min_beta*output_scaler_scale) + output_scaler_bias
y_BLL_min_beta = (np.array(y_BLL_min_beta)).squeeze()

y_std_BLL = (np.array(y_std_BLL)).squeeze()

mse_y = mean_squared_error([y_RNN_den[-1]], [y_BLL_den[-1]])
mse_Theta0 = mean_squared_error([Uo[0][0]], [W_BLL[-1,0]])
mse_Theta1 = mean_squared_error([Uo[0][1]], [W_BLL[-1,1]])
mse_Theta2 = mean_squared_error([Uo[0][2]], [W_BLL[-1,2]])
mse_Theta3 = mean_squared_error([Uo[0][3]], [W_BLL[-1,3]])
mse_Theta4 = mean_squared_error([Uo[0][4]], [W_BLL[-1,4]])
mse_Theta5 = mean_squared_error(bo[0], [W_BLL[-1,5]])

time = np.arange(0, N)
hour_ticks = np.array([0, 48, 96, 144, 192, 240, 288])
hour_labels = np.array([0, 4, 8, 12, 16, 20, 24]) 

# ========== INPUT PLOT ==========
pyplt.figure()
pyplt.plot(time, U_applied_den, linewidth=4)
pyplt.ylabel('Temperature [°C]', fontsize=44)
pyplt.xlim((0, N))
pyplt.xlabel('Time [h]', fontsize=44)
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=44)  
pyplt.subplots_adjust(bottom=0.18)
pyplt.show()

# ========== OUTPUT + CONFIDENCE INTERVAL ==========
pyplt.figure()
pyplt.plot(time, y_RNN_den, linewidth=4)
pyplt.plot(time, y_BLL_den, linewidth=4)
pyplt.plot(time, Y_ref_den, color='black', linestyle='--', linewidth=4)
pyplt.fill_between(time, y_BLL_min_beta, y_BLL_max_beta, color='orange', alpha=0.3)
pyplt.plot(time, 70*np.ones((N,1)), color='black', linewidth=4)
pyplt.plot(time, 90*np.ones((N,1)), color='black', linewidth=4)
pyplt.ylabel('Temperature [°C]', fontsize=44)
pyplt.xlabel('Time [h]', fontsize=44)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=44)  
pyplt.subplots_adjust(bottom=0.18)
pyplt.show()

# ========== UNCERTAINTY ==========
pyplt.figure()
pyplt.plot(time, beta_n * y_std_BLL, linewidth=4)
pyplt.plot(time, epsilon_sigma * np.ones((N,1)), color='red', linestyle='--', linewidth=4)
pyplt.ylabel(r'$w$', fontsize=44)
pyplt.xlabel('Time [h]', fontsize=44)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=44)  
pyplt.subplots_adjust(bottom=0.18)
pyplt.show()

# ========== COST FUNCTIONS ==========
pyplt.figure()
# pyplt.plot(time, J_pess, label=r'$J^p$', linewidth=4)
# pyplt.plot(time, J_opt, label=r'$J^{o, \epsilon}$', linewidth=4)
pyplt.plot(time, J_pess - J_opt, linewidth=4)
pyplt.plot(time, 2*epsilon_sigma*H*Lips*np.ones((N,1)), linestyle='--', color='red', linewidth=4)
pyplt.ylabel('Cost', fontsize=44)
pyplt.xlabel('Time [h]', fontsize=44)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=44)  
pyplt.subplots_adjust(bottom=0.18)
pyplt.show()

# ========== WEIGHTS ==========
for i in range(5):
    pyplt.figure()
    pyplt.plot(time, Uo[0][i] * np.ones((N,1)), linewidth=4)
    pyplt.plot(time, W_BLL[:, i], linewidth=4)
    pyplt.fill_between(
        time,
        W_BLL[:, i] - beta_n * W_std_BLL[:, i],
        W_BLL[:, i] + beta_n * W_std_BLL[:, i],
        color='orange', alpha=0.3
    )
    pyplt.ylabel(f'Weight {i}', fontsize=44)
    pyplt.xlabel('Time [h]', fontsize=44)
    pyplt.xlim((0, N))
    pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
    pyplt.tick_params(axis='both', labelsize=44)  
    pyplt.subplots_adjust(bottom=0.18)
    pyplt.subplots_adjust(left=0.15)
    pyplt.show()

# ========== BIAS ==========
pyplt.figure()
pyplt.plot(time, bo * np.ones((N,1)), linewidth=4)
pyplt.plot(time, W_BLL[:, 5], linewidth=4)
pyplt.fill_between(
    time,
    W_BLL[:, 5] - beta_n * W_std_BLL[:, 5],
    W_BLL[:, 5] + beta_n * W_std_BLL[:, 5],
    color='orange', alpha=0.3
)
pyplt.ylabel('Bias', fontsize=44)
pyplt.xlabel('Time [h]', fontsize=44)
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=44)  
pyplt.subplots_adjust(bottom=0.18)
pyplt.show()


# threshold = 0.000001
# for n in range (0,N):
#     if np.all(slack_matrix[n, :] > threshold): # np.any(slack_matrix[n, :] > threshold):
#         pypyplt.plot(np.arange(0, H), slack_matrix[n,:], label=f'It ={n}')
# pypyplt.legend(fontsize=44)
# pypyplt.ylabel('Slack', fontsize=44)
# pypyplt.xlim((0, H))
# pypyplt.show()

# ax[3].plot(time, (W_std_BLL[:, 0]))
# ax[3].plot(time, (W_std_BLL[:, 1]))
# ax[3].plot(time, (W_std_BLL[:, 2]))
# ax[3].plot(time, (W_std_BLL[:, 3]))
# ax[3].plot(time, (W_std_BLL[:, 4]))
# ax[3].plot(time, (W_std_BLL[:, 5]))
# ax[3].set_ylabel('Theta_std', fontsize=44)
# ax[3].set_xlim((0, N))
# pypyplt.show()

# pypyplt.plot(time, xT_L_x_all, label="xT Lambda_inv x")
# pypyplt.legend(fontsize=6)
# pypyplt.xlim((0, N))
# pypyplt.show()

# pypyplt.plot(time, U_pess, label='U_pess')
# pypyplt.plot(time, U_explo, label='U_explo')
# pypyplt.plot(time, U_applied_den, label='U_applied')
# pypyplt.legend(fontsize=6)
# pypyplt.ylabel('J', fontsize=44)
# pypyplt.xlim((0, N))
# pypyplt.show()
