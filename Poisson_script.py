import torch
import numpy as np
from torch.autograd import grad
from tqdm import tqdm

sigmas = [0.3, 0.4, 0.5]
d_Ns = [5, 10, 20, 30, 40, 50, 70, 100, 150, 200]
iterations = 20000
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

min_xyt, max_xyt = 0, 1

torch.manual_seed(0)
np.random.seed(0)

# The source function
def f(x1, x2):
   return 4 * torch.ones_like(x1)

# The corresponding solution
def u_sol(x1, x2):
   return x1 - torch.square(x1) + x2 - torch.square(x2)

x_coll_tr, y_coll_tr = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 150),
          torch.linspace(min_xyt, max_xyt, 150),
          indexing = "xy"
      )
x_coll_tr = x_coll_tr.to(dev)
y_coll_tr = y_coll_tr.to(dev)

x_coll_ts, y_coll_ts = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 150),
          torch.linspace(min_xyt, max_xyt, 150),
          indexing = "xy"
      )
x_coll_ts = x_coll_ts.to(dev)
y_coll_ts = y_coll_ts.to(dev)

x1_tr = x_coll_tr.unsqueeze(-1).detach().requires_grad_()
x2_tr = y_coll_tr.unsqueeze(-1).detach().requires_grad_()

x_tr = torch.stack((x1_tr,x2_tr), dim=-1)
x_br_tr = x_tr[(
  (x_tr[..., 0] == 0) |
  (x_tr[..., 0] == 1) |
  (x_tr[..., 1] == 0) |
  (x_tr[..., 1] == 1)
)].detach()


x1_ts = x_coll_ts.unsqueeze(-1).detach().requires_grad_()
x2_ts = y_coll_ts.unsqueeze(-1).detach().requires_grad_()

x_ts = torch.stack((x1_ts,x2_ts), dim=-1)
x_br_ts = x_ts[(
  (x_ts[..., 0] == 0) |
  (x_ts[..., 0] == 1) |
  (x_ts[..., 1] == 0) |
  (x_ts[..., 1] == 1)
)].detach()

br_sol_tr = u_sol(x_br_tr[...,0],x_br_tr[...,1])
br_sol_ts = u_sol(x_br_ts[...,0],x_br_ts[...,1])

num_sig = len(sigmas)
noise_trs = {}
noise_tss = {}
for sig in sigmas:
  noise_trs[sig] = torch.empty_like(br_sol_tr).uniform_(-np.sqrt(3)*sig, np.sqrt(3)*sig).to(dev).to(torch.float32)
  noise_tss[sig] = torch.empty_like(br_sol_ts).uniform_(-np.sqrt(3)*sig, np.sqrt(3)*sig).to(dev).to(torch.float32)


class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x


lambda_0 = 1.1
# lambda_s = 1
lr = 1e-2
supervision_loss_choice = torch.nn.MSELoss()

def compute_pde_loss(N, x1, x2, create_graph=True):
  u = N(torch.cat((x1, x2),dim=-1))
  ones = torch.ones_like(u)
  u_x1 = grad (u, x1, ones, create_graph = True, retain_graph=True)[0]
  u_x2 = grad (u, x2, ones, create_graph = True, retain_graph=True)[0]
  ones = torch.ones_like(u_x1)
  u_x1x1 = grad(u_x1, x1, ones, create_graph = create_graph, retain_graph=True)[0]
  u_x2x2 = grad(u_x2, x2, ones, create_graph = create_graph, retain_graph=create_graph)[0]

  # Compute the loss for the PDE
  return (f(x1, x2) + u_x1x1+ u_x2x2).square().mean()

def train(N, sig):
  train_loss = np.zeros(iterations)
  test_loss = np.zeros(int(iterations / 100))
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr)

  br_noisy_tr = (br_sol_tr + noise_trs[sig])
  br_noisy_ts = (br_sol_ts + noise_tss[sig])

  for i in tqdm(range(iterations)):

    optimizer.zero_grad()

    # Denoting by u the realization function of the ANN, compute
    u_b_tr = N(x_br_tr)
    # Compute the loss for the noisy boundary condition
    br_loss_tr = (u_b_tr - br_noisy_tr).square().mean()

    # Compute the loss for the PDE
    pde_loss_tr = compute_pde_loss(N, x1_tr, x2_tr, create_graph=True)

    # Compute the total loss and perform a gradient step
    train = pde_loss_tr + lambda_0 * br_loss_tr
    train_loss[i] = train.detach().cpu().item()
    train.backward()
    optimizer.step()
    if i % 100 == 0:
      u_b_ts = N(x_br_ts)
      # Compute the loss for the noisy boundary condition
      br_loss_ts = (u_b_ts - br_noisy_ts).square().mean()

      # Compute the loss for the PDE
      pde_loss_ts = compute_pde_loss(N, x1_ts, x2_ts, create_graph=False)
      test = pde_loss_ts + lambda_0 * br_loss_ts
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
    NN = TwoLayerNN(2,d_Ns[k],1).to(dev)
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
  torch.save(net_dict, f"weights/Poisson_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}.pt")
  np.savez(f"results/Poisson_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}",train=train_errors,test=test_errors)