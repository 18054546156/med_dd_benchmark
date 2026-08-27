import torch

real = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])

rr = torch.cdist(real,real,p=2)

print(rr)