import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from tqdm import tqdm


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

min_xyt, max_xyt = 0, 1
sigmas = [0.4]
densities = np.array([16])
d_Ns = [2, 3, 4, 5, 6, 8, 10, 12, 20, 30, 40]

lr = 1e-3

torch.manual_seed(0)


x_coll, y_coll, t_coll = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, 32),
          torch.linspace(min_xyt, max_xyt, 32),
          torch.linspace(0, max_xyt - min_xyt, 32),
          indexing = "xy"
      )
x_coll = x_coll.to(dev)
y_coll = y_coll.to(dev)
t_coll = t_coll.to(dev)

# The initial condition
def g_in(x):
    return x.square().sum(axis=-1, keepdims=True)

# The corresponding solution
def u_sol(xt):
    x_in = xt[:,:2]
    t_in = xt[:,2]
    op = 1 + 4 * (t_in)
    return torch.add(torch.div(x_in.square().sum(axis=1), op), np.log(op)).reshape(-1,1)

def generate_data_split(sigmas, densities):
  Ns = densities ** 3
  train_split = np.floor(0.8 * Ns).astype(int)

  X_train = [[0]*len(densities)]*len(sigmas)
  y_train = [[0]*len(densities)]*len(sigmas)
  X_test = [[0]*len(densities)]*len(sigmas)
  y_test = [[0]*len(densities)]*len(sigmas)
  for i in range(len(sigmas)):
    for j in range(len(densities)):
      tr_xinput, tr_yinput, tr_tinput = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, densities[j]),
          torch.linspace(min_xyt, max_xyt, densities[j]),
          torch.linspace(0, max_xyt - min_xyt, densities[j]),
          indexing = "xy"
      )
      tr_input = torch.stack([tr_xinput, tr_yinput, tr_tinput], dim=-1).reshape(-1, 3)

      tr_output = u_sol(tr_input.cpu())

      X_train[i][j] = tr_input[:train_split[j],:].to(dev).to(torch.float32)
      X_test[i][j] = tr_input[train_split[j]:,:].to(dev)

      y_train[i][j] = (tr_output[:train_split[j]] + np.random.normal(0, sigmas[i], (train_split[j], 1))).to(dev).to(torch.float32)
      y_test[i][j] = tr_output[train_split[j]:].to(dev)
  return X_train, y_train, X_test, y_test

def generate_true_data(densities):
  X = [0]*len(densities)
  y = [0]*len(densities)
  for j in range(len(densities)):
    tr_xinput, tr_yinput, tr_tinput = torch.meshgrid(
        torch.linspace(min_xyt, max_xyt, densities[j]),
        torch.linspace(min_xyt, max_xyt, densities[j]),
        torch.linspace(0, max_xyt - min_xyt, densities[j]),
        indexing = "xy"
    )
    tr_input = torch.stack([tr_xinput, tr_yinput, tr_tinput], dim=-1).reshape(-1, 3)

    tr_output = u_sol(tr_input.cpu())

    X[j] = tr_input.to(dev).to(torch.float32)

    y[j] = tr_output.to(dev).to(torch.float32)
  return X, y


num_sig = len(sigmas)

num_den = densities.shape[0]
X_train, y_train, X_test, y_test = generate_data_split(sigmas, densities)
X_true, y_true = generate_true_data(2 * densities)


class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x

J = 2560 # the batch size
iterations = 20000
lambda_0 = 0.3
lambda_s = 0.5
supervision_loss_choice = torch.nn.MSELoss()

def train(N, X_train, y_train, X_test, y_test, X_true, y_true):
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr, weight_decay=0)
  training_losses = np.zeros(iterations)
  testing_losses = np.zeros(iterations)
  true_losses = np.zeros(iterations)
  for i in tqdm(range(iterations)):
    x1, x2, t = x_coll.unsqueeze(-1), y_coll.unsqueeze(-1), t_coll.unsqueeze(-1)
    x = torch.cat([x1,x2],axis=-1)
    x1.requires_grad_()
    x2.requires_grad_()
    t.requires_grad_()

    optimizer.zero_grad()

    # Denoting by u the realization function of the ANN, compute
    # u(0, x) for each x in the batch
    u0 = N(torch.cat((x,torch.zeros_like(t)),axis=-1))
    # Compute the loss for the initial condition
    initial_loss = (u0 - g_in(x)).square().mean()

    # Compute the partial derivatives using automatic differentiation
    u = N(torch.cat((x1, x2, t),axis=-1))
    ones = torch.ones_like(u)
    u_t = grad(u, t, ones, create_graph = True)[0]
    u_x1 = grad (u, x1, ones, create_graph = True)[0]
    u_x2 = grad (u, x2, ones, create_graph = True)[0]
    ones = torch.ones_like(u_x1)
    u_x1x1 = grad(u_x1, x1, ones, create_graph = True)[0]
    u_x2x2 = grad(u_x2, x2, ones, create_graph = True)[0]

    # Compute the loss for the PDE
    Laplace = u_x1x1 + u_x2x2
    pde_loss = (u_t - Laplace + u_x1**2 + u_x2**2).square().mean()

    # Compute the supervision loss
    supervision_loss_train = supervision_loss_choice(N(X_train), y_train)
    supervision_loss_test = supervision_loss_choice(N(X_test), y_test)

    # Compute the total loss and perform a gradient step
    train_loss = pde_loss + lambda_0*initial_loss + lambda_s * supervision_loss_train
    training_losses[i] = train_loss
    test_loss = pde_loss + lambda_0*initial_loss + lambda_s * supervision_loss_test
    testing_losses[i] = test_loss
    true_losses[i] = supervision_loss_choice(N(X_true), y_true)
    train_loss.backward()
    optimizer.step()
  return training_losses, testing_losses, true_losses

num_dN = len(d_Ns)
second_layer_weight_mean = 3
second_layer_weight_std = 1

train_errors = np.zeros((num_sig,num_den,num_dN, iterations))
test_errors = np.zeros((num_sig,num_den,num_dN, iterations))
true_errors = np.zeros((num_sig,num_den,num_dN, iterations))

result = np.zeros((num_sig,num_den,num_dN),dtype=bool)
for i in range(num_sig):
  for j in range(num_den):
    for k in range(num_dN):
      NN = TwoLayerNN(3,d_Ns[k],1).to(dev)
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
      train_errors[i,j,k,:], test_errors[i,j,k,:], true_errors[i,j,k,:] = train(NN, X_train[i][j], y_train[i][j], X_test[i][j], y_test[i][j], X_true[j], y_true[j])
      result[i,j,k] = (train_errors[i,j,k,iterations-1] <= sigmas[i]**2)

np.savez(f"results/HJB_results_sigmas_{sigmas}_densities_{densities}_dN_{d_Ns}_lr_{lr}.npz",train_errors=train_errors,test_errors=test_errors,true_errors=true_errors)