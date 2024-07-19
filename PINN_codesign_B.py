# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class PINN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, num_layer):
        super().__init__()
        layers = [nn.Linear(dim_input, dim_hidden), nn.Tanh()]
        for i in range(num_layer - 1):
            layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.Tanh()])
        layers.extend([nn.Linear(dim_hidden, dim_output), nn.Softplus()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def sample_uniform_sphere(d, num, radius, device):
    samples = torch.randn(num, d, device=device)  # 正規分布からサンプリング
    samples = samples / samples.norm(dim=1, keepdim=True)  # 半径1の円周上に正規化
    radii = torch.pow(torch.rand(num, 1, device=device), 1.0 / d) * radius  # 円内部へのスケーリング
    samples = samples * radii
    return samples


#############################################################################
# Parameter settings
#############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Problem setting
d = 10  # dimension
p = 1
q = 1
R = 1
gamma = 1

# Neural network
dim_input = d + 1  # NNの入力の次元(状態の次元と同じ)
dim_output = 1  # NNの出力の次元（価値関数がスカラーなので常に1）
dim_hidden = 64  # 隠れ層のユニット数
num_layer = 3  # 中間層の数
model = PINN(dim_input, dim_output, dim_hidden, num_layer).to(device)
learning_rate = 1e-4  # NNの学習率
betas = (0.9, 0.9999)  # Adamのmomentum (0.9, 0.999)
eps = 1e-06  # 1e-08
optimizer_model = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps)


beta = torch.tensor([1.00], requires_grad=True, device=device)
learning_rate_beta = 0.01  # betaの学習率
optimizer_beta = optim.Adam([beta], lr=learning_rate_beta)

# Hyper parameters
num_pde_samples = 1000  # 領域内部のサンプル数
num_boundary = 1000  # 領域境界のサンプル数
num_origin = 1  # 原点でのサンプル数
epochs = 40000

mu_pde = 1e-0  # PDEロスの重み
mu_boundary = 1  # 境界ロスの重み
mu_origin = 1  # 原点ロスの重み
radius = 2  # 境界条件の半径

 # Logs
LOG_DIR = f'logs/d={d}/mupde={mu_pde:.1e}-lr={learning_rate:.1e}_' + datetime.now().strftime('%m%d%H%M')
summary_writer = SummaryWriter(LOG_DIR)
'$tensorboard --logdir logs'

#data save
loss_history = []
beta_history = []

######################################################################
# Training loop
######################################################################
for epoch in range(epochs):
    
    #params
    k = (math.sqrt((q * gamma) ** 2 + 4 * p * q * beta) - gamma * q) / (2 * beta ** 2)
    #############################
    # PDE loss
    #############################
    # Random samples for x
    x = sample_uniform_sphere(d, num_pde_samples, radius, device)
    beta_noise = beta.expand(num_pde_samples, 1) + torch.rand(num_pde_samples, 1, device=device) * 0.1
    state = torch.cat((x, beta_noise), dim=1)
    state.requires_grad_(True)

    # Calculate V and gradient
    V = model(state)
    dV_dx = torch.autograd.grad(V.sum(), state, create_graph=True)[0]
    dV_dx_sub = dV_dx[:, :d]

    # Calculate Laplasian
    Laplasian = 0
    for i in range(d):
        d2V_dxi2 = torch.autograd.grad(dV_dx_sub[:, i].sum(), state, retain_graph=True)[0]
        d2V_dxi2 = d2V_dxi2[:, i].unsqueeze(1)
        Laplasian += d2V_dxi2
        
    
    # Loss term
    w = Laplasian - (beta ** 2) * (dV_dx_sub ** 2).sum(dim=1, keepdim=True) / (2 * q) \
        + p * (x ** 2).sum(dim=1, keepdim=True) \
        + (beta ** 2) * (dV_dx_sub ** 2).sum(dim=1, keepdim=True) / (4 * q) \
        - 2 * k * d \
        - gamma * V

    loss_w = torch.mean(w ** 2)

    #############################
    # Boundary loss
    #############################
    # origin
    x_b0 = torch.zeros(1, d, device=device)
    beta_b0 = beta.expand(1, 1) + torch.rand(1, 1, device=device) * 0.1
    state_b0 = torch.cat((x_b0, beta_b0), dim=1)
    V_b0 = model(state_b0)
    loss_b0 = torch.mean(V_b0 ** 2)

    # circumference
    x_bR = torch.randn(num_boundary, d, device=device)
    x_bR = x_bR / x_bR.norm(dim=1, keepdim=True) * 2  # radius R=2
    beta_bR = beta.expand(num_boundary, 1) + torch.rand(num_boundary, 1, device=device) * 0.1
    state_bR = torch.cat((x_bR, beta_bR), dim=1)
    V_bR = model(state_bR)
    y_bR = k * (x_bR ** 2).sum(dim=1, keepdim=True)
    loss_bR = torch.nn.functional.mse_loss(V_bR, y_bR)

    ###############################################
    # Calculate Total loss and update
    ###############################################

    loss = mu_pde * loss_w + mu_boundary * loss_bR + mu_origin * loss_b0
    loss.backward()
    optimizer_model.step()
    optimizer_model.zero_grad()
    
    ###############################################
    # beta update
    ###############################################
    if epoch % 500 == 0:
        V_beta = model(state)
        loss_beta = torch.mean(beta_noise) + 4 * torch.mean(V_beta)
        loss_beta.backward()
        optimizer_beta.step()
        optimizer_beta.zero_grad()


    # #data save
    loss_history.append(loss.item())
    beta_history.append(beta.item())
   
    
    #############################################
    # Logs
    #############################################
    summary_writer.add_scalar("00_Model Loss", loss, epoch)
    summary_writer.add_scalar("01_PDE Loss", loss_w, epoch)
    summary_writer.add_scalar("02_Boundary Loss", loss_bR, epoch)
    summary_writer.flush()

    if epoch % 1000 == 0:
         print(f'Epoch {epoch}, Model Loss: {loss.item()}, J: {loss_beta.item()}, beta: {beta.item()}')
         ckpt_path = LOG_DIR + f'/pinn-{epoch}'
         torch.save({'weights': model.state_dict(),
                     'loss': loss},
                    ckpt_path)
#############################################
# Plot
#############################################    
# Loss plot
plt.figure(figsize=(14, 8))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Beta plot
plt.figure(figsize=(14, 8))
plt.plot(beta_history, label='Beta')
plt.xlabel('Epoch')
plt.ylabel('Beta')
plt.show()

# V plot
x_values = np.linspace(-radius, radius, 1000)
x_zeros = np.zeros((1000, d-1))
x_plot = np.hstack((x_values.reshape(-1, 1), x_zeros))
beta_plot = beta.expand(1000, 1).detach().cpu().numpy()
state_plot = np.hstack((x_plot, beta_plot))
state_plot_tensor = torch.tensor(state_plot, device=device, dtype=torch.float32)
V_plot = model(state_plot_tensor).detach().cpu().numpy()
V_coreect = k * x_values ** 2
plt.figure(figsize=(14, 8))
plt.plot(x_values, V_plot, label='V PINNs')
plt.plot(x_values, V_coreect, label='V correct')
plt.xlabel('x1')
plt.ylabel('V')
plt.legend()
plt.show()
