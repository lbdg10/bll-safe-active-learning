#%% Import libraries
import casadi as ca
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as pyplt
import random
import time
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
H = 24

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

# Pesi iniziali
Uo_mpc = layers[1][0]["weights"][0][0]["weight"][0]
bo_mpc = layers[1][0]["weights"][0][0]["bias"][0].T

# Pesi con prezzi 1 e 4
# Uo_mpc = np.array([[-0.79299545,  1.24694443, -1.03558731,  0.60805035, -0.358284,  0.48060226], [-0.01830006, -0.03972816, -0.36888409,  0.16631031, -0.73983908, -0.34461594]], dtype=np.float32)
# bo_mpc = np.array([[-0.32545066], [-0.1361928 ]], dtype=np.float32)

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

# Weights of the cost function
w_Tdiff = 1.6
w_U = 5 # 3
w_y_slack = 0.01

# MPC initialization
x0 = np.random.uniform(-1, 1, size=n_x)
u0_scaled = (80 - input_scaler_bias[0][0]) / input_scaler_scale[0][0]
U_prev = np.ones((n_u, H)) * u0_scaled
X_prev = 2 * np.random.rand(n_x, H + 1) - 1
U_applied = []
y_RNN = []
n = 0
t_exe = np.zeros(N)

# Rule-based strategy
Rulebased = 0
U_Rulebased = np.full(N, 80)
U_Rulebased[:84] = 80
U_Rulebased[252:] = 80
U_Rulebased = (U_Rulebased - input_scaler_bias[0][0]) / input_scaler_scale[0][0]


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
    ykp = Uo_mpc @ xkp + bo_mpc
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

# MPC
def mpc_economic(x_n, U_prev, X_prev, n):

    # Casadi setup and optimization variables
    opti = ca.Opti()
    U = opti.variable(n_u, H)
    X = opti.variable(n_x, H+1)
    epigrafica = opti.variable(1, 1)
    slack_y = opti.variable(n_y, H)

    # Initial values
    opti.subject_to(X[:,0] == x_n)

    # Boundary constraints
    opti.subject_to(U <= u_max)
    opti.subject_to(U >= u_min)
    
    for i in range(n_x):
        opti.subject_to(X[i,:] <= x_max)
        opti.subject_to(X[i,:] >= x_min)
        
    # Slack variables non negativity
    for i in range(n_y):
        opti.subject_to(slack_y[i,:] >= 0)

    # Needed variables
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

        # Output constraints
        opti.subject_to(y_pred_RNN >= y_min_variable_ca[:,n+h])
        opti.subject_to(y_pred_RNN <= y_max_ca)
        # opti.subject_to(y_pred_RNN >= y_min_variable_ca[:,n+h] - slack_y[:,h])
        # opti.subject_to(y_pred_RNN <= y_max_ca + slack_y[:,h])

        # Cost function: electrical price and slack
        if h==0:
            J_step += current_price_ca[h] * y_pred_RNN[1] + w_y_slack * ca.sumsqr(slack_y[:,h]) + w_U*(U[:,h] - U_prev[:,0])**2
        else:
            J_step += current_price_ca[h] * y_pred_RNN[1] + w_y_slack * ca.sumsqr(slack_y[:,h]) + w_U*(U[:,h] - U[:,h-1])**2

    # Terminal set
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
    opti.subject_to(y_pred_RNN[0]-Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(-y_pred_RNN[0]+Y_ref_ca[0,H-1] <= epigrafica)
    opti.subject_to(epigrafica >= 0)
    J_step += w_Tdiff*epigrafica

    # Minimize
    opti.minimize(J_step)

    # Warm initialization
    opti.set_initial(U, U_prev)
    opti.set_initial(X, X_prev)
    opti.set_initial(slack_y, 0)
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

    return U_opt, U_opt_tot, X_opt_tot



################################################ Simulation ################################################

while n < N:
    
    if Rulebased:
        U_applied.append(U_Rulebased[n])
        y0, x0 = RNN_model_1step(x0, U_applied[-1], d_const)
        y_RNN.append(y0)

    else:
        t0 = time.perf_counter()
        U_mpc_economic, U_prev, X_prev = mpc_economic(x0, U_prev, X_prev, n)
        t_exe[n] = time.perf_counter() - t0
        U_prev[:, :-1] = U_prev[:, 1:] 
        U_prev[:, -1] = U_prev[:, -1]
        X_prev[:, :-1] = X_prev[:, 1:] 
        X_prev[:, -1] = X_prev[:, -1] 

        U_applied.append(np.array(U_mpc_economic).squeeze())
        y0, x0 = RNN_model_1step(x0, U_applied[-1], d_const)
        y_RNN.append(y0)


    n += 1


################################################ Plot ################################################

# Denormalize variables
U_applied = np.array(U_applied)
U_applied_den = (U_applied*input_scaler_scale[0][0]) + input_scaler_bias[0][0]
U_applied_den = (np.array(U_applied_den)).squeeze()

y_RNN_arr = np.array(y_RNN)  
y_RNN_arr = y_RNN_arr[:, :, 0]
y_RNN_den = (y_RNN_arr*output_scaler_scale) + output_scaler_bias

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

# ========== OUTPUT 1 ==========
pyplt.figure()
pyplt.plot(time, y_RNN_den[:,0], linewidth=5, color='tab:blue')
pyplt.plot(time, y_min_T_variable[0:N], color='black', linewidth=5)
pyplt.plot(time, (y_max_T_value)*np.ones((N,1)), color='black', linewidth=5)
pyplt.ylabel(r'$T_5^s$ [°C]', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.ylim((y_min_T_variable.min()*0.99,y_max_T_value*1.01))
pyplt.xlim((0, N))
pyplt.xticks(ticks=hour_ticks, labels=hour_labels)
pyplt.tick_params(axis='both', labelsize=60)  
pyplt.subplots_adjust(bottom=0.26)
pyplt.subplots_adjust(left=0.18)
pyplt.show()

# ========== OUTPUT 2 ==========
pyplt.figure()
pyplt.plot(time, y_RNN_den[:,1]/1000000, linewidth=5, color='tab:blue')
pyplt.plot(time, y_min_P_value[0:N]/1000000, color='black', linewidth=5)
pyplt.plot(time, y_max_P_value/1000000*np.ones((N,1)), color='black', linewidth=5)
pyplt.ylabel(r'$P_0$ [MW]', fontsize=60)
pyplt.xlabel(r'Time [h]', fontsize=60)
pyplt.ylim((y_min_P_value.min()/1000000*0.99, y_max_P_value/1000000*1.01))
pyplt.xlim((0, N))
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


print("Average MPC time per step: ", np.mean(t_exe))
print("Maximum MPC time per step: ", np.max(t_exe))


##################### Save results #####################
time = np.asarray(time).reshape(-1, 1)
U_applied_den = np.asarray(U_applied_den).reshape(-1, 1)
y_RNN_den = np.asarray(y_RNN_den)

data = {
    "time": time,
    "U_applied_den": U_applied_den,
    "y_RNN_den": y_RNN_den,
    "y_min_T_variable": y_min_T_variable,
    "y_max_T_value": y_max_T_value,
    "y_min_P_value": y_min_P_value,
    "y_max_P_value": y_max_P_value
}

savemat("Benchmark_solution.mat", data)