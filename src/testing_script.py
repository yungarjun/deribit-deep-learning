import torch

# 1) Load your tensor
surf = torch.load("btc_surfaces.pt")   # shape [T, N_nodes]

# 2) Check overall min/max
print("Min price:", surf.min().item())
print("Max price:", surf.max().item())

# 3) Count exact zeros
num_zeros = (surf == 0).sum().item()
total    = surf.numel()
print(f"Zero entries: {num_zeros} out of {total} "
      f"({100*num_zeros/total:.2f}%)")

# 4) Check for NaNs
num_nans = torch.isnan(surf).sum().item()
print(f"NaN entries: {num_nans}")


print(surf.shape)