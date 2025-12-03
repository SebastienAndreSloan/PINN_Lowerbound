import torch
import numpy as np
from torch.autograd import grad
from tqdm import tqdm

iterations = 3000
sigmas = [0.05, 0.06, 0.07, 0.08, 0.09]
d_Ns = [2, 8, 22, 44, 58, 92, 134, 158]
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

min_xyt, max_xyt = 0, 1
nu = 0.01
rho = 1

N_r = 50 # cubed
N_0 = 50 # squared
N_t = 60 # cubed
N_0_t = 60 # squared

torch.manual_seed(0)

x_coll_r, y_coll_r, t_coll_r = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_r),
          torch.linspace(min_xyt, max_xyt, N_r),
          torch.linspace(0, max_xyt - min_xyt, N_r),
          indexing = "xy"
      )
coll_r = torch.stack([x_coll_r,y_coll_r,t_coll_r],dim=-1).reshape(-1,3).to(dev)

x_coll_0, y_coll_0 = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_0),
          torch.linspace(min_xyt, max_xyt, N_0),
          indexing = "xy"
      )
coll_0 = torch.stack([x_coll_0, y_coll_0],dim=-1).reshape(-1,2).to(dev)

x_coll_test, y_coll_test, t_coll_test= torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_t),
          torch.linspace(min_xyt, max_xyt, N_t),
          torch.linspace(0, max_xyt - min_xyt, N_t),
          indexing = "xy"
      )
coll_test = torch.stack([x_coll_test,y_coll_test,t_coll_test],dim=-1).reshape(-1,3).to(dev)

x_coll_0_test, y_coll_0_test = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_0_t),
          torch.linspace(min_xyt, max_xyt, N_0_t),
          indexing = "xy"
      )
coll_0_test = torch.stack([x_coll_0_test,y_coll_0_test],dim=-1).reshape(-1,2).to(dev)

def g_in(x,y):
    u = torch.sin(x) * torch.cos(y)
    v = -1 * torch.cos(x) * torch.sin(y)
    return torch.stack((u,v),dim=-1)

# The corresponding solution
def u_sol(x,y,t):
    u = torch.sin(x) * torch.cos(y) * torch.exp(-2 * nu * t)
    v = -1 * torch.cos(x) * torch.sin(y) * torch.exp(-2 * nu * t)
    p = (rho / 4) * (torch.cos(2*x) + torch.cos(2*y)) * torch.exp(-4 * nu * t)
    return torch.stack((u,v,p),dim=-1)

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

      tr_output = u_sol(tr_xinput.cpu(),tr_yinput.cpu(),tr_tinput.cpu()).reshape(-1, 3)

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

    tr_output = u_sol(tr_xinput.cpu(),tr_yinput.cpu(),tr_tinput.cpu()).reshape(-1, 3)

    X[j] = tr_input.to(dev).to(torch.float32)

    y[j] = tr_output.to(dev).to(torch.float32)
  return X, y

num_sig = len(sigmas)
densities = np.array([16])
num_den = densities.shape[0]
X_train, Y_train, X_test, Y_test = generate_data_split(sigmas, densities)
class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x

batch_no = 100
test_rate = 100
lambda_0 = 0.3
lambda_1 = 1
lr = 1e-3
lambda_s = 0.5
supervision_loss_choice = torch.nn.MSELoss()


batsz_r = N_r**3 // batch_no
batsz_0 = N_0**2 // batch_no
batsz_test = N_t**3 // batch_no
batsz_0_test = N_0_t**2 // batch_no

def train(N, X_train, Y_train, X_test, Y_test):
  train_loss = np.zeros(iterations)
  test_loss = np.zeros(iterations // test_rate)
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr)
  for i in tqdm(range(iterations)):
  # for i in range(iterations):
    indices_r = torch.randperm(N_r**3)
    indices_0 = torch.randperm(N_0**2)
    indices_test = torch.randperm(N_t**3)
    indices_0_test = torch.randperm(N_0_t**2)

    for j in range(0,batch_no):
      batch_idx_r = indices_r[j*batsz_r:(j+1)*batsz_r]
      batch_idx_0 = indices_0[j*batsz_0:(j+1)*batsz_0]

      x_r, y_r, t_r = coll_r[batch_idx_r,0], coll_r[batch_idx_r,1], coll_r[batch_idx_r,2]
      x_r.requires_grad_(True)
      y_r.requires_grad_(True)
      t_r.requires_grad_(True)

      x_0, y_0 = coll_0[batch_idx_0,0],coll_0[batch_idx_0,1]
      x_0.requires_grad_(True)
      y_0.requires_grad_(True)
      
      optimizer.zero_grad()

      # Compute the partial derivatives using automatic differentiation
      out = N(torch.stack((x_r, y_r, t_r),dim=-1))
      u=out[...,0]
      v=out[...,1]
      p=out[...,2]

      u_x = grad(u, x_r, torch.ones_like(u), create_graph = True)[0].squeeze() # du/dx
      u_y = grad(u, y_r, torch.ones_like(u), create_graph = True)[0].squeeze() # du/dy
      u_t = grad(u, t_r, torch.ones_like(u), create_graph = True)[0].squeeze()  # du/dt
      v_x = grad(v, x_r, torch.ones_like(v), create_graph = True)[0].squeeze() # dv/dx
      v_y = grad(v, y_r, torch.ones_like(v), create_graph = True)[0].squeeze() # dv/dy
      v_t = grad(v, t_r, torch.ones_like(u), create_graph = True)[0].squeeze()  # dv/dt
      p_x = grad(p, x_r, torch.ones_like(p), create_graph = True)[0].squeeze() # dp/dx
      p_y = grad(p, y_r, torch.ones_like(p), create_graph = True)[0].squeeze() # dp/dy

      u_xx = grad(u_x, x_r, torch.ones_like(u_x), create_graph = True)[0].squeeze() # d2u/dx2
      u_yy = grad(u_y, y_r, torch.ones_like(u_y), create_graph = True)[0].squeeze() # d2u/dy2
      v_xx = grad(v_x, x_r, torch.ones_like(v_x), create_graph = True)[0].squeeze() # d2v/dx2
      v_yy = grad(v_y, y_r, torch.ones_like(v_y), create_graph = True)[0].squeeze() # d2v/dy2

      x_mom = u_t + (u * u_x) + (v * u_y) + ((1 / rho) * p_x) - (nu * (u_xx + u_yy)) # x momentum
      y_mom = v_t + (u * v_x) + (v * v_y) + ((1 / rho) * p_y) - (nu * (v_xx + v_yy)) # y momentum
      cont = u_x + v_y # continuity

      x_mom_loss = torch.nn.MSELoss()(x_mom, torch.zeros_like(x_mom)) # x momentum loss
      y_mom_loss = torch.nn.MSELoss()(y_mom, torch.zeros_like(y_mom)) # y momentum loss
      cont_loss = torch.nn.MSELoss()(cont, torch.zeros_like(cont)) # continuity loss

      u0 = N(torch.stack((x_0,y_0,torch.zeros_like(x_0)),dim=-1))[...,:2]
      initial_loss = (u0 - g_in(x_0,y_0)).square().mean()

      # Compute the PDE loss
      pde_loss = (x_mom_loss + y_mom_loss + lambda_1 * cont_loss).square().mean()

      supervision_loss_train = supervision_loss_choice(N(X_train), Y_train)

      if i % test_rate == 0:
        batch_idx_test = indices_test[j*batsz_test:(j+1)*batsz_test]
        batch_idx_0_test = indices_0_test[j*batsz_0_test:(j+1)*batsz_0_test]
        N.eval()
        x_test, y_test, t_test = coll_test[batch_idx_test,0], coll_test[batch_idx_test,1], coll_test[batch_idx_test,2]
        x_test.requires_grad_(True)
        y_test.requires_grad_(True)
        t_test.requires_grad_(True)

        x_0_test, y_0_test = coll_0_test[batch_idx_0_test,0],coll_0_test[batch_idx_0_test,1]
        x_0_test.requires_grad_(True)
        y_0_test.requires_grad_(True)

        u0_test = N(torch.stack((x_0_test,y_0_test,torch.zeros_like(x_0_test)),dim=-1))[...,:2]
        initial_loss_test = (u0_test - g_in(x_0_test,y_0_test)).square().mean()

        out_test = N(torch.stack((x_test, y_test, t_test),dim=-1))
        u_test=out_test[...,0]
        v_test=out_test[...,1]
        p_test=out_test[...,2]

        u_x_test = grad(u_test, x_test, torch.ones_like(u_test), create_graph = True)[0].squeeze() # du/dx
        u_y_test = grad(u_test, y_test, torch.ones_like(u_test), create_graph = True)[0].squeeze() # du/dy
        u_t_test = grad(u_test, t_test, torch.ones_like(u_test), create_graph = True)[0].squeeze()  # du/dt
        v_x_test = grad(v_test, x_test, torch.ones_like(v_test), create_graph = True)[0].squeeze() # dv/dx
        v_y_test = grad(v_test, y_test, torch.ones_like(v_test), create_graph = True)[0].squeeze() # dv/dy
        v_t_test = grad(v_test, t_test, torch.ones_like(u_test), create_graph = True)[0].squeeze()  # dv/dt
        p_x_test = grad(p_test, x_test, torch.ones_like(p_test), create_graph = True)[0].squeeze() # dp/dx
        p_y_test = grad(p_test, y_test, torch.ones_like(p_test), create_graph = True)[0].squeeze() # dp/dy

        u_xx_test = grad(u_x_test, x_test, torch.ones_like(u_x_test), create_graph = True)[0].squeeze() # d2u/dx2
        u_yy_test = grad(u_y_test, y_test, torch.ones_like(u_y_test), create_graph = True)[0].squeeze() # d2u/dy2
        v_xx_test = grad(v_x_test, x_test, torch.ones_like(v_x_test), create_graph = True)[0].squeeze() # d2v/dx2
        v_yy_test = grad(v_y_test, y_test, torch.ones_like(v_y_test), create_graph = True)[0].squeeze() # d2v/dy2

        x_mom_test = u_t_test + (u_test * u_x_test) + (v_test * u_y_test) + ((1 / rho) * p_x_test) - (nu * (u_xx_test + u_yy_test)) # x momentum
        y_mom_test = v_t_test + (u_test * v_x_test) + (v_test * v_y_test) + ((1 / rho) * p_y_test) - (nu * (v_xx_test + v_yy_test)) # y momentum
        cont_test = u_x_test + v_y_test # continuity

        x_mom_loss_test = torch.nn.MSELoss()(x_mom_test, torch.zeros_like(x_mom_test)) # x momentum loss
        y_mom_loss_test = torch.nn.MSELoss()(y_mom_test, torch.zeros_like(y_mom_test)) # y momentum loss
        cont_loss_test = torch.nn.MSELoss()(cont_test, torch.zeros_like(cont_test)) # continuity loss

        # Compute the PDE loss
        pde_loss_test = (x_mom_loss_test + y_mom_loss_test + lambda_1 * cont_loss_test).square().mean()

        supervision_loss_test = supervision_loss_choice(N(X_test), Y_test)

        test = pde_loss_test + lambda_0*initial_loss_test + lambda_s * supervision_loss_test
        test_loss[i // test_rate] = test
        N.train()

      # Compute the total loss and perform a gradient step
      train = pde_loss + lambda_0*initial_loss + lambda_s * supervision_loss_train
      train_loss[i] = train
      train.backward()
      optimizer.step()

  return train_loss, test_loss

num_dN = len(d_Ns)
num_den = 1
second_layer_weight_mean = 0
second_layer_weight_std = 1

train_errors = np.zeros((num_sig, num_den,num_dN,iterations))
test_errors = np.zeros((num_sig, num_den,num_dN,iterations // 100))
net_dict = {}

result = np.zeros((num_den,num_dN),dtype=bool)
for i in range(num_sig):
  for j in range(num_den):
    for k in range(num_dN):
      NN = 0
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
      train_errors[i,j,k,:], test_errors[i,j,k,:] = train(NN, X_train[i][j], Y_train[i][j], X_test[i][j], Y_test[i][j])
      net_dict[f"net_dN_{d_Ns[k]}"] = NN.state_dict()
      print("Succefully done net number", k)
  print("Succefully scanned sigma number", i)
  torch.save(net_dict, f"weights_NS_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}.pt")
  np.savez(f"result_NS_dNs_{d_Ns}_s_{sigmas[i]}_i_{iterations}",train=train_errors,test=test_errors)