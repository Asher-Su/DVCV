import torch

targets = [torch.tensor(2),torch.tensor(3),torch.tensor(8)]
targets_tensor = torch.stack(targets)
sort = torch.argsort(targets_tensor,descending=True)
sort = sort.tolist()
print(sort)