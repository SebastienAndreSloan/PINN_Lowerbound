import torch
import numpy as np
from torch.autograd import grad
from tqdm import tqdm

sigmas = [0.3, 0.4, 0.5]
d_Ns = [5, 10, 20, 30, 40, 50, 70, 100, 150, 200]
iterations = 10000
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

min_xyt, max_xyt = 0, 1
nu = 0.001
rho = 1

torch.manual_seed(0)
np.random.seed(0)

# The corresponding solution
def u_sol(x1, x2, t):
   u = torch.sin(x1) * torch.cos(x2) * torch.exp(-2 * nu * t)
   v = -1 * torch.cos(x1) * torch.sin(x2) * torch.exp(-2 * nu * t)
   p = rho / 4 * (torch.cos(2 * x1) + torch.cos(2 * x2)) * torch.exp(-4 * nu * t)
   return torch.stack((u,v,p), dim=-1)

x_coll_tr, y_coll_tr, t_coll_tr = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 20),
          torch.linspace(min_xyt, max_xyt, 20),
          torch.linspace(0, max_xyt - min_xyt, 20),
          indexing = "xy"
      )
x_coll_tr = x_coll_tr.to(dev)
y_coll_tr = y_coll_tr.to(dev)
t_coll_tr = t_coll_tr.to(dev)

x_sup_tr, y_sup_tr, t_sup_tr = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 16),
          torch.linspace(min_xyt, max_xyt, 16),
          torch.linspace(0, max_xyt - min_xyt, 16),
          indexing = "xy"
      )
x_sup_tr = x_sup_tr.to(dev)
y_sup_tr = y_sup_tr.to(dev)
t_sup_tr = t_sup_tr.to(dev)

x_coll_ts, y_coll_ts, t_coll_ts = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 50),
          torch.linspace(min_xyt, max_xyt, 50),
          torch.linspace(0, max_xyt - min_xyt, 50),
          indexing = "xy"
      )
x_coll_ts = x_coll_ts.to(dev)
y_coll_ts = y_coll_ts.to(dev)
t_coll_ts = t_coll_ts.to(dev)

x_sup_ts, y_sup_ts, t_sup_ts = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 16),
          torch.linspace(min_xyt, max_xyt, 16),
          torch.linspace(0, max_xyt - min_xyt, 16),
          indexing = "xy"
      )
x_sup_ts = x_sup_ts.to(dev)
y_sup_ts = y_sup_ts.to(dev)
t_sup_ts = t_sup_ts.to(dev)

x1_tr = x_coll_tr.unsqueeze(-1).detach().requires_grad_()
x2_tr = y_coll_tr.unsqueeze(-1).detach().requires_grad_()
t_tr = t_coll_tr.unsqueeze(-1).detach().requires_grad_()
x1_sup_tr = x_sup_tr.unsqueeze(-1).detach().requires_grad_()
x2_sup_tr = y_sup_tr.unsqueeze(-1).detach().requires_grad_()
t_sup_tr = t_sup_tr.unsqueeze(-1).detach().requires_grad_()

x_tr = torch.stack((x1_tr,x2_tr,t_tr), dim=-1)
x_sup_tr = torch.stack((x1_sup_tr,x2_sup_tr,t_sup_tr), dim=-1).detach()
x_in_tr = x_tr[(
  (x_tr[..., 2] == 0)
)].detach()

x1_ts = x_coll_ts.unsqueeze(-1).detach().requires_grad_()
x2_ts = y_coll_ts.unsqueeze(-1).detach().requires_grad_()
t_ts = t_coll_ts.unsqueeze(-1).detach().requires_grad_()
x1_sup_ts = x_sup_ts.unsqueeze(-1).detach().requires_grad_()
x2_sup_ts = y_sup_ts.unsqueeze(-1).detach().requires_grad_()
t_sup_ts = t_sup_ts.unsqueeze(-1).detach().requires_grad_()

x_ts = torch.stack((x1_ts,x2_ts,t_ts), dim=-1)
x_sup_ts = torch.stack((x1_sup_ts,x2_sup_ts,t_sup_ts), dim=-1).detach()
x_in_ts = x_ts[(
  (x_ts[..., 2] == 0)
)].detach()


in_sol_tr = u_sol(x_in_tr[...,0],x_in_tr[...,1],x_in_tr[...,2])
in_sol_ts = u_sol(x_in_ts[...,0],x_in_ts[...,1],x_in_ts[...,2])
sup_sol_tr = u_sol(x1_sup_tr,x2_sup_tr,t_sup_tr).detach()
sup_sol_ts = u_sol(x1_sup_ts,x2_sup_ts,t_sup_ts).detach()



num_sig = len(sigmas)
noise_trs = {}
noise_tss = {}

for sig in sigmas:
  noise_trs[sig] = torch.empty_like(sup_sol_tr).uniform_(-np.sqrt(3)*sig, np.sqrt(3)*sig).to(dev).to(torch.float32)
  noise_tss[sig] = torch.empty_like(sup_sol_ts).uniform_(-np.sqrt(3)*sig, np.sqrt(3)*sig).to(dev).to(torch.float32)


class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x


lambda_0 = 0.3
lambda_s = 1.1
lr = 1e-2
supervision_loss_choice = torch.nn.MSELoss()

def compute_pde_loss(N, x1, x2, t, create_graph=True):
  u, v, p = N(torch.cat((x1, x2, t),dim=-1)).unbind(dim=-1)
  ones = torch.ones_like(u)
  u_x1 = grad (u, x1, ones, create_graph = True, retain_graph=True)[0]
  u_x2 = grad (u, x2, ones, create_graph = True, retain_graph=True)[0]
  u_t = grad (u, t, ones, create_graph = True, retain_graph=True)[0]
  ones = torch.ones_like(v)
  v_x1 = grad (v, x1, ones, create_graph = True, retain_graph=True)[0]
  v_x2 = grad (v, x2, ones, create_graph = True, retain_graph=True)[0]
  v_t = grad (v, t, ones, create_graph = True, retain_graph=True)[0]
  ones = torch.ones_like(p)
  p_x1 = grad (p, x1, ones, create_graph = True, retain_graph=True)[0]
  p_x2 = grad (p, x2, ones, create_graph = True, retain_graph=True)[0]
  ones = torch.ones_like(u_x1)
  u_x1x1 = grad(u_x1, x1, ones, create_graph = create_graph, retain_graph=True)[0]
  u_x2x2 = grad(u_x2, x2, ones, create_graph = create_graph, retain_graph=True)[0]
  ones = torch.ones_like(v_x1)
  v_x1x1 = grad(v_x1, x1, ones, create_graph = create_graph, retain_graph=True)[0]
  v_x2x2 = grad(v_x2, x2, ones, create_graph = create_graph, retain_graph=create_graph)[0]

  # Compute the loss for the PDE
  x_mom = u_t + (u * u_x1) + (v * u_x2) + ((1 / rho) * p_x1) - (nu * (u_x1x1 + u_x2x2)) # x momentum
  y_mom = v_t + (u * v_x1) + (v * v_x2) + ((1 / rho) * p_x2) - (nu * (v_x1x1 + v_x2x2)) # y momentum
  cont = u_x1 + v_x2 # continuity

  return x_mom.square().mean() + y_mom.square().mean() + cont.square().mean()

def train(N, sig):
  train_loss = np.zeros(iterations)
  test_loss = np.zeros(int(iterations / 100))
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr)

  sup_noisy_tr = (sup_sol_tr + noise_trs[sig])
  sup_noisy_ts = (sup_sol_ts + noise_tss[sig])
  for i in tqdm(range(iterations)):

    optimizer.zero_grad()

    # Denoting by u the realization function of the ANN, compute
    u_in_tr = N(x_in_tr)
    # Compute the loss for the noisy boundary condition
    in_loss_tr = (u_in_tr - in_sol_tr).square().mean()

    u_sup_tr = N(x_sup_tr)
    sup_loss_tr = (u_sup_tr - sup_noisy_tr).square().mean()

    # Compute the loss for the PDE
    pde_loss_tr = compute_pde_loss(N, x1_tr, x2_tr, t_tr, create_graph=True)

    # Compute the total loss and perform a gradient step
    train = pde_loss_tr + lambda_0 * in_loss_tr + lambda_s * sup_loss_tr
    train_loss[i] = train.detach().cpu().item()
    train.backward()
    optimizer.step()
    if i % 100 == 0:
      u_in_ts = N(x_in_ts)
      # Compute the loss for the noisy boundary condition
      in_loss_ts = (u_in_ts - in_sol_ts).square().mean()

      u_sup_ts = N(x_sup_ts)
      sup_loss_ts = (u_sup_ts - sup_noisy_ts).square().mean()

      # Compute the loss for the PDE
      pde_loss_ts = compute_pde_loss(N, x1_ts, x2_ts, t_ts, create_graph=False)
      test = pde_loss_ts + lambda_0 * in_loss_ts + lambda_s * sup_loss_ts
      test_loss[i // 100] = test.detach().cpu().item()
  return train_loss, test_loss


num_dN = len(d_Ns)
second_layer_weight_mean = 0
second_layer_weight_std = 1

train_errors = np.zeros((num_sig,num_dN,iterations))
test_errors = np.zeros((num_sig,num_dN,iterations // 100))
net_dict = {}

from pathlib import Path

Path("results").mkdir(parents=True, exist_ok=True)
Path("weights").mkdir(parents=True, exist_ok=True)

result = np.zeros((num_sig,num_dN),dtype=bool)
for i in range(num_sig):
  for k in range(num_dN):
    NN = TwoLayerNN(3,d_Ns[k],3).to(dev)
    for param in NN.layer2.parameters():
        param.requires_grad = False
    for name, param in NN.named_parameters():
      if "bias" in name:
          param.data.fill_(0)
    for name, param in NN.named_parameters():
      if "bias" in name:
          param.requires_grad = False
    with torch.no_grad():
        NN.layer2.weight.copy_(torch.normal(second_layer_weight_mean, second_layer_weight_std, size=NN.layer2.weight.shape))
    print(f"Starting net number {k} with width {d_Ns[k]} at sigma {sigmas[i]}")
    train_errors[i,k,:], test_errors[i,k, :] = train(NN, sigmas[i])
    net_dict[f"net_dN_{d_Ns[k]}"] = NN.state_dict()
    print("Succefully done net number", k)
  print("Succefully scanned sigma number", i)
  torch.save(net_dict, f"weights/NS_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}.pt")
  np.savez(f"results/NS_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}",train=train_errors,test=test_errors)