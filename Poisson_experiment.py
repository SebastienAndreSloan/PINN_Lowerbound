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

x_coll, y_coll = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 150),
          torch.linspace(min_xyt, max_xyt, 150),
          indexing = "xy"
      )
x_coll = x_coll.to(dev)
y_coll = y_coll.to(dev)

# The boundary condition
def u_br(x):
    return torch.zeros_like(x)

# The source function
def f(x1, x2):
   return 4 * torch.ones_like(x1)

# The corresponding solution
def u_sol(x1, x2):
   return x1 - torch.square(x1) + x2 - torch.square(x2)


num_sig = len(sigmas)
densities = np.array([100])
num_den = densities.shape[0]

class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x


lambda_0 = 1
lambda_s = 1
lr = 1e-2
supervision_loss_choice = torch.nn.MSELoss()

def train(N):
  train_loss = np.zeros(iterations)
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr)
  for i in tqdm(range(iterations)):

    x1, x2 = x_coll.unsqueeze(-1), y_coll.unsqueeze(-1)
    x1.requires_grad_()
    x2.requires_grad_()

    x = torch.stack((x1,x2), dim=-1)
    x_br = x[(
      (x[..., 0] == 0) |
      (x[..., 0] == 1) |
      (x[..., 1] == 0) |
      (x[..., 1] == 1)
    )]

    optimizer.zero_grad()

    # Denoting by u the realization function of the ANN, compute
    # u(0, x) for each x in the batch
    u_b = N(x_br)
    # Compute the loss for the noisy boundary condition
    br_sol = u_sol(x_br[...,0],x_br[...,1])
    br_noisy = (br_sol + torch.normal(torch.zeros_like(br_sol), sigmas[0])).to(dev).to(torch.float32)
    br_loss = (u_b - br_noisy).square().mean()

    # Compute the partial derivatives using automatic differentiation
    u = N(torch.cat((x1, x2),axis=-1))
    ones = torch.ones_like(u)
    u_x1 = grad (u, x1, ones, create_graph = True)[0]
    u_x2 = grad (u, x2, ones, create_graph = True)[0]
    ones = torch.ones_like(u_x1)
    u_x1x1 = grad(u_x1, x1, ones, create_graph = True)[0]
    u_x2x2 = grad(u_x2, x2, ones, create_graph = True)[0]

    # Compute the loss for the PDE
    pde_loss = (f(x1, x2) + u_x1x1 + u_x2x2).square().mean()

    # Compute the total loss and perform a gradient step
    train = pde_loss + lambda_0 * br_loss
    train_loss[i] = train
    test_loss = pde_loss + lambda_0 * br_loss
    train.backward()
    optimizer.step()
  training_loss = train_loss
  testing_loss = test_loss
  return training_loss, testing_loss


num_dN = len(d_Ns)
second_layer_weight_mean = 0
second_layer_weight_std = 1

train_errors = np.zeros((num_sig,num_den,num_dN,iterations))
test_errors = np.zeros((num_sig,num_den,num_dN))
net_dict = {}

result = np.zeros((num_sig,num_den,num_dN),dtype=bool)
for i in range(num_sig):
  for j in range(num_den):
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
      train_errors[i,j,k,:], test_errors[i,j,k] = train(NN)
      net_dict[f"net_dN_{d_Ns[k]}"] = NN.state_dict()
      print("Succefully done net number", k)
  print("Succefully scanned sigma number", i)
  torch.save(net_dict, f"weights_newPoisson_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}.pt")
  np.savez(f"result_new_Poisson_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}",train=train_errors,test=test_errors)