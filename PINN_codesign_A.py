import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden):
        super().__init__()
        layers = [nn.Linear(dim_input, dim_hidden), nn.Tanh()]
        layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.Tanh()])
        layers.extend([nn.Linear(dim_hidden, dim_output), nn.Softplus()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

dim_input = 3
dim_output = 1
dim_hidden = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PINN(dim_input, dim_output, dim_hidden).to(device)
rho = torch.tensor([1.00], requires_grad=True, device=device)
rho_history = []
loss_history = []
epochs = 150000

optimizer_model = optim.Adam(model.parameters(), lr=0.003)
optimizer_rho = optim.Adam([rho], lr=0.05)

# Training loop
for epoch in range(epochs):
    x1 = torch.rand(1000, 1, device=device) * 2 - 1
    x2 = torch.rand(1000, 1, device=device) * 2 - 1
    rho_noise = rho.expand(1000, 1) + torch.rand(1000, 1, device=device) * 0.1
    state = torch.cat((x1, x2, rho_noise), dim=1)
    state.requires_grad_(True)
    
    V = model(state)
    dV_dx = torch.autograd.grad(V.sum(), state, create_graph=True)[0]
    dV_dx1 = dV_dx[:, 0].reshape(-1, 1)
    dV_dx2 = dV_dx[:, 1].reshape(-1, 1)
    
    f1 = -x1 ** 3 - x2
    f2 = x1 + x2
    W = dV_dx1 * f1 + dV_dx2 * f2 + x1 ** 2 + x2 ** 2 - 0.25 * ((rho_noise * dV_dx2) ** 2)
    loss_W = torch.mean(W ** 2)
    
    # Boundary loss
    x1_b = torch.zeros(100, 1, device=device)
    x2_b = torch.zeros(100, 1, device=device)
    rho_b = rho.expand(100, 1) + torch.rand(100, 1, device=device) * 0.1
    boundary_state = torch.cat((x1_b, x2_b, rho_b), dim=1)
    V_boundary = model(boundary_state)
    loss_B = torch.mean((V_boundary - torch.zeros(1, device=device)) ** 2)
    
    # Total loss and model update
    loss = loss_W + loss_B
    loss_history.append(loss.item())
    loss.backward()
    optimizer_model.step()
    optimizer_model.zero_grad()
    
    if epoch % 500 == 0:
        V_rho = model(state)
        loss_rho = torch.mean(rho_noise) + 4 * torch.mean(V_rho)
        loss_rho.backward()
        optimizer_rho.step()
        optimizer_rho.zero_grad()
    
    rho_history.append(rho.item())
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Model Loss: {loss.item()}, J : {loss_rho.item()}, Rho: {rho.item()}')
        

# Plotting
plt.figure()
plt.plot(range(epochs), rho_history)
plt.xlabel('Epochs')
plt.ylabel('rho')
plt.show()

plt.plot(range(epochs), loss_history)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()

x2_np = np.linspace(-1, 1, 100).reshape(-1, 1)
x1_np = np.zeros_like(x2_np)
rho_np = rho.expand(100, 1).detach().cpu().numpy()
input_np = np.hstack([x1_np, x2_np, rho_np])
input_tensor = torch.tensor(input_np, dtype=torch.float32, device=device)
V_pred = model(input_tensor).detach().cpu().numpy()
plt.plot(x2_np, V_pred)
plt.xlabel('x2')
plt.ylabel('V')
plt.show()
